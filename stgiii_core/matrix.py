"""
Matrix generation for StageIII Simulator (v1.0)

全組み合わせセルの評価値（pIC50）を生成・保持

生成モデル（要約）:
- Main effects（Free-Wilsonで説明可能）
- Pairwise interactions（smooth + spike(activity cliff)）
- Residual error（heavy-tail）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .exceptions import UniqueArgmaxError
from .indexer import CellIndexer


def _double_center(table: NDArray[np.float64]) -> NDArray[np.float64]:
    """行・列平均を0にする（二重中心化）。"""
    row_mean = table.mean(axis=1, keepdims=True)
    col_mean = table.mean(axis=0, keepdims=True)
    grand_mean = table.mean()
    return table - row_mean - col_mean + grand_mean


def _standardize_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """1D配列を標準化（平均0・標準偏差1）。std=0なら0配列。"""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0:
        return np.zeros_like(values)
    return (values - mean) / std


def _standardize_2d(table: NDArray[np.float64]) -> NDArray[np.float64]:
    """2D配列を標準化（平均0・標準偏差1）。std=0なら0配列。"""
    mean = float(np.mean(table))
    std = float(np.std(table))
    if std <= 0:
        return np.zeros_like(table)
    return (table - mean) / std


def _orthonormal_columns(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int
) -> NDArray[np.float64]:
    """乱数行列の列をQRで正規直交化して返す。"""
    mat = rng.normal(0.0, 1.0, size=(n_rows, n_cols))
    q, _r = np.linalg.qr(mat)
    return q[:, :n_cols].astype(np.float64, copy=False)


@dataclass
class Matrix:
    """全組み合わせセルの評価値を保持するデータ構造（v1.0）"""

    # スコア
    y_latent: NDArray[np.float64]              # 潜在スコア（Top-1判定基準）
    y_obs: NDArray[np.float64]                 # 観測スコア（clip後）

    # 分解成分（デバッグ/検証用）
    y_main: NDArray[np.float64]
    y_int: NDArray[np.float64]
    y_int_smooth: NDArray[np.float64]
    y_int_spike: NDArray[np.float64]
    y_res: NDArray[np.float64]

    # 生成パラメータの“実体”（必要最小限）
    global_bias: float
    main_effects: List[NDArray[np.float64]]  # slot -> (N_s,)
    bb_embeddings: List[NDArray[np.float64]]  # slot -> (N_s, d)
    interaction_tables: Dict[Tuple[int, int], NDArray[np.float64]]  # (s,t)->(N_s,N_t)

    # 難易度サマリ
    variance_fractions: Dict[str, float]
    clip_rate: float

    # ランキング
    top1_index: int
    topk_indices: NDArray[np.int64]

    @property
    def n_total(self) -> int:
        """総セル数"""
        return len(self.y_latent)

    def get_y_obs(self, index: int) -> float:
        """指定セルの観測値を取得"""
        return float(self.y_obs[index])

    def get_y_latent(self, index: int) -> float:
        """指定セルの潜在スコアを取得"""
        return float(self.y_latent[index])

    def is_top1(self, index: int) -> bool:
        """指定セルがTop-1か判定（latent基準）"""
        return index == self.top1_index

    def is_topk(self, index: int) -> bool:
        """指定セルがTop-kに含まれるか判定（latent基準）"""
        return index in self.topk_indices


class MatrixGenerator:
    """Matrix生成器（v1.0）"""

    def __init__(
        self,
        config: SimulationConfig,
        indexer: CellIndexer,
        rng: np.random.Generator
    ) -> None:
        self.config = config
        self.indexer = indexer
        self.rng = rng

    def generate(self) -> Matrix:
        """
        Matrixを生成

        Raises:
            UniqueArgmaxError: max_matrix_regeneration回の再生成でも
                               argmaxが一意にならない場合
        """
        for _attempt in range(1, self.config.max_matrix_regeneration + 1):
            matrix = self._generate_single()
            if matrix is not None:
                return matrix

        raise UniqueArgmaxError(self.config.max_matrix_regeneration)

    def _generate_single(self) -> Matrix | None:
        n_total = self.indexer.n_total
        n_slots = self.config.n_slots

        obs_low, obs_high = self.config.obs_clip_range
        mu_target = self.config.global_bias
        sigma_target_base = (obs_high - obs_low) / 6.0

        # 全セルの座標を取得
        all_indices = np.arange(n_total, dtype=np.int64)
        all_coords = self.indexer.batch_indices_to_coords(all_indices)

        # 1) BB埋め込み（シリーズ混合）
        d = self.config.embedding_dim
        series_div = self.config.embedding_series_divisor
        sigma_z = self.config.embedding_sigma

        bb_embeddings: List[NDArray[np.float64]] = []
        for slot_idx, slot_config in enumerate(self.config.slots):
            n_bb = slot_config.n_building_blocks
            k_s = int(np.ceil(n_bb / series_div))

            centers = self.rng.normal(0.0, 1.0, size=(k_s, d))

            # 均等にシリーズ割当（近いBBが必ずしも近いわけではないが、系列の存在を再現）
            group_ids = np.arange(n_bb) % k_s
            self.rng.shuffle(group_ids)

            noise = self.rng.normal(0.0, sigma_z, size=(n_bb, d))
            z = centers[group_ids] + noise
            bb_embeddings.append(z.astype(np.float64, copy=False))

        # 2) 主作用（従来方針: 一様分布、後段で寄与比率スケーリング）
        main_effects: List[NDArray[np.float64]] = []
        for slot_config in self.config.slots:
            n_bb = slot_config.n_building_blocks
            m = self.rng.uniform(-1.0, 1.0, size=n_bb).astype(np.float64, copy=False)
            main_effects.append(m)

        y_main = np.zeros(n_total, dtype=np.float64)
        for slot_idx in range(n_slots):
            bb_idx = all_coords[:, slot_idx]
            y_main += main_effects[slot_idx][bb_idx]

        # 3) 相互作用（pairwise）
        dmat = np.array(self.config.get_slot_distance_matrix(), dtype=np.float64)
        lam = self.config.distance_lambda

        eta = float(self.config.eta_spike)
        smooth_coef = float(np.sqrt(max(0.0, 1.0 - eta)))
        spike_coef = float(np.sqrt(max(0.0, eta)))

        r = self.config.interaction_rank
        h = self.config.spike_hotspots
        nu_spike = self.config.spike_nu
        sigma_spike = self.config.spike_sigma
        ell = self.config.spike_ell

        y_int_smooth = np.zeros(n_total, dtype=np.float64)
        y_int_spike = np.zeros(n_total, dtype=np.float64)
        interaction_tables: Dict[Tuple[int, int], NDArray[np.float64]] = {}

        for s in range(n_slots):
            for t in range(s + 1, n_slots):
                scale = float(np.exp(-dmat[s, t] / lam))

                z_s = bb_embeddings[s]
                z_t = bb_embeddings[t]
                n_s, _d1 = z_s.shape
                n_t, _d2 = z_t.shape

                # smooth (low-rank bilinear): I_smooth = (Z_s U diag(w)) (Z_t V)^T
                u = _orthonormal_columns(self.rng, d, r)
                v = _orthonormal_columns(self.rng, d, r)
                w = self.rng.normal(0.0, 1.0, size=r).astype(np.float64, copy=False)

                a = z_s @ u  # (n_s, r)
                b = z_t @ v  # (n_t, r)
                a_w = a * w  # (n_s, r)
                i_smooth = (a_w @ b.T).astype(np.float64, copy=False)  # (n_s, n_t)

                # spike (hotspots on embedding space)
                i_spike = np.zeros((n_s, n_t), dtype=np.float64)
                if h > 0:
                    for _ in range(h):
                        i0 = int(self.rng.integers(0, n_s))
                        j0 = int(self.rng.integers(0, n_t))

                        c_a = z_s[i0]
                        c_b = z_t[j0]
                        amp = float(self.rng.standard_t(nu_spike) * sigma_spike)

                        d_a = np.sum((z_s - c_a) ** 2, axis=1)
                        d_b = np.sum((z_t - c_b) ** 2, axis=1)
                        k_a = np.exp(-d_a / (2.0 * (ell ** 2)))
                        k_b = np.exp(-d_b / (2.0 * (ell ** 2)))
                        i_spike += amp * (k_a[:, None] * k_b[None, :])

                # double-center then standardize each component (eta_spike as "variance-ish" knob)
                i_smooth_dc = _double_center(i_smooth)
                i_spike_dc = _double_center(i_spike)
                i_smooth_hat = _standardize_2d(i_smooth_dc)
                i_spike_hat = _standardize_2d(i_spike_dc)

                i_pair = scale * (smooth_coef * i_smooth_hat + spike_coef * i_spike_hat)
                interaction_tables[(s, t)] = i_pair

                # add to per-cell interaction components
                idx_s = all_coords[:, s]
                idx_t = all_coords[:, t]
                if smooth_coef != 0.0:
                    y_int_smooth += scale * smooth_coef * i_smooth_hat[idx_s, idx_t]
                if spike_coef != 0.0:
                    y_int_spike += scale * spike_coef * i_spike_hat[idx_s, idx_t]

        y_int = y_int_smooth + y_int_spike

        # 4) 残差（heavy-tail）
        y_res = self.rng.standard_t(self.config.residual_nu, size=n_total).astype(
            np.float64, copy=False
        )

        # 5) 成分を標準化して、分散比率（f_*)で合成
        y_main_hat = _standardize_1d(y_main)
        y_int_hat = _standardize_1d(y_int)
        y_res_hat = _standardize_1d(y_res)

        w_main = float(np.sqrt(self.config.f_main))
        w_int = float(np.sqrt(self.config.f_int))
        w_res = float(np.sqrt(self.config.f_res))

        c_main = w_main * y_main_hat
        c_int = w_int * y_int_hat
        c_res = w_res * y_res_hat

        y_latent_raw = c_main + c_int + c_res

        # 6) 出力スケール調整（obsレンジ内に収まりやすくする）
        y0 = _standardize_1d(y_latent_raw)

        # clip率の上限に合わせてσを縮める（過剰clipを避ける）
        sigma_target = float(sigma_target_base)
        if self.config.clip_rate_max <= 0:
            sigma_target = 0.0
        elif self.config.clip_rate_max < 1.0:
            def clip_rate_for_sigma(sig: float) -> float:
                y_tmp = mu_target + y0 * sig
                return float(np.mean((y_tmp < obs_low) | (y_tmp > obs_high)))

            initial_clip = clip_rate_for_sigma(sigma_target)
            if initial_clip > self.config.clip_rate_max:
                lo = 0.0
                hi = sigma_target
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    if clip_rate_for_sigma(mid) <= self.config.clip_rate_max:
                        lo = mid
                    else:
                        hi = mid
                sigma_target = lo

        y_latent = mu_target + y0 * sigma_target
        clip_rate = float(np.mean((y_latent < obs_low) | (y_latent > obs_high)))
        y_obs = np.clip(y_latent, obs_low, obs_high)

        # argmaxの一意性チェック（latent基準）
        max_val = np.max(y_latent)
        max_indices = np.where(np.isclose(y_latent, max_val))[0]
        if len(max_indices) > 1:
            return None

        top1_index = int(max_indices[0])
        topk_k = self.config.topk_k
        topk_indices = np.argsort(y_latent)[-topk_k:][::-1].astype(np.int64)

        # サマリ（分散寄与）
        var_main = float(np.var(c_main))
        var_int = float(np.var(c_int))
        var_res = float(np.var(c_res))
        var_sum = var_main + var_int + var_res
        if var_sum <= 0:
            var_frac = {"main": 0.0, "int": 0.0, "res": 0.0}
        else:
            var_frac = {
                "main": var_main / var_sum,
                "int": var_int / var_sum,
                "res": var_res / var_sum,
            }

        var_int_smooth = float(np.var(y_int_smooth))
        var_int_spike = float(np.var(y_int_spike))
        var_int_sum = var_int_smooth + var_int_spike
        if var_int_sum > 0:
            var_frac["int_smooth"] = var_int_smooth / var_int_sum
            var_frac["int_spike"] = var_int_spike / var_int_sum
        else:
            var_frac["int_smooth"] = 0.0
            var_frac["int_spike"] = 0.0

        return Matrix(
            y_latent=y_latent,
            y_obs=y_obs,
            y_main=y_main,
            y_int=y_int,
            y_int_smooth=y_int_smooth,
            y_int_spike=y_int_spike,
            y_res=y_res,
            global_bias=mu_target,
            main_effects=main_effects,
            bb_embeddings=bb_embeddings,
            interaction_tables=interaction_tables,
            variance_fractions=var_frac,
            clip_rate=clip_rate,
            top1_index=top1_index,
            topk_indices=topk_indices,
        )
