"""
Matrix generation for StageIII Simulator

全組み合わせセルの評価値（pIC50）を生成・保持
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .indexer import CellIndexer
from .exceptions import UniqueArgmaxError


@dataclass
class Matrix:
    """全組み合わせセルの評価値を保持するデータ構造"""

    y_true: NDArray[np.float64]                # 真値配列（1D, 長さ n_total）
    y_obs: NDArray[np.float64]                 # 観測値配列（clipped, 1D, 長さ n_total）
    main_effects: List[NDArray[np.float64]]    # 各スロットの主作用（slot_bias込み）
    global_bias: float                         # グローバルバイアス
    slot_biases: NDArray[np.float64]           # 各スロットのslot_bias
    errors: NDArray[np.float64]                # 各セルの誤差項（1D, 長さ n_total）
    top1_index: int                            # 正解セル（argmax(y_true)）のインデックス
    topk_indices: NDArray[np.int64]            # Top-kセルのインデックス配列

    @property
    def n_total(self) -> int:
        """総セル数"""
        return len(self.y_true)

    @property
    def max_y_true(self) -> float:
        """y_trueの最大値"""
        return float(self.y_true[self.top1_index])

    @property
    def max_y_obs(self) -> float:
        """正解セルのy_obs値"""
        return float(self.y_obs[self.top1_index])

    def get_y_obs(self, index: int) -> float:
        """指定セルの観測値を取得"""
        return float(self.y_obs[index])

    def get_y_true(self, index: int) -> float:
        """指定セルの真値を取得"""
        return float(self.y_true[index])

    def is_top1(self, index: int) -> bool:
        """指定セルがTop-1か判定"""
        return index == self.top1_index

    def is_topk(self, index: int) -> bool:
        """指定セルがTop-kに含まれるか判定"""
        return index in self.topk_indices


class MatrixGenerator:
    """Matrix生成器"""

    def __init__(
        self,
        config: SimulationConfig,
        indexer: CellIndexer,
        rng: np.random.Generator
    ) -> None:
        """
        Args:
            config: シミュレーション設定
            indexer: セルインデックス変換器
            rng: 乱数生成器
        """
        self.config = config
        self.indexer = indexer
        self.rng = rng

    def generate(self) -> Matrix:
        """
        Matrixを生成

        Returns:
            生成されたMatrix

        Raises:
            UniqueArgmaxError: max_matrix_regeneration回の再生成でも
                               argmaxが一意にならない場合
        """
        for attempt in range(1, self.config.max_matrix_regeneration + 1):
            matrix = self._generate_single()
            if matrix is not None:
                return matrix

        raise UniqueArgmaxError(self.config.max_matrix_regeneration)

    def _generate_single(self) -> Matrix | None:
        """
        単一のMatrix生成を試行

        Returns:
            成功時はMatrix、argmaxが一意でない場合はNone
        """
        n_total = self.indexer.n_total

        # グローバルバイアス: 固定値
        global_bias = self.config.global_bias

        # スロットバイアス: 各スロットにUniform(-0.5, 0.5)
        n_slots = self.config.n_slots
        slot_biases = self.rng.uniform(
            *self.config.slot_bias_range,
            size=n_slots
        )

        # 主作用（各スロット）: Uniform(main_low, main_high) + slot_bias
        main_low, main_high = self.config.main_effect_range
        main_effects: List[NDArray[np.float64]] = []

        for slot_idx, slot_config in enumerate(self.config.slots):
            n_bb = slot_config.n_building_blocks
            # 生の主作用
            raw_main = self.rng.uniform(main_low, main_high, size=n_bb)
            # slot_biasを加算
            main_with_bias = raw_main + slot_biases[slot_idx]
            main_effects.append(main_with_bias)

        # 誤差: Normal(0, σ_gen^2) をclip
        err_low, err_high = self.config.error_clip_range
        sigma_gen = self.config.sigma_gen
        errors_raw = self.rng.normal(0, sigma_gen, size=n_total)
        errors = np.clip(errors_raw, err_low, err_high)

        # y_trueの計算: bias + Σmain_effect + error
        y_true = self._compute_y_true(
            global_bias, main_effects, errors
        )

        # argmaxの一意性チェック
        max_val = np.max(y_true)
        max_indices = np.where(np.isclose(y_true, max_val))[0]
        if len(max_indices) > 1:
            # 同値最大が複数 -> 再生成
            return None

        top1_index = int(max_indices[0])

        # Top-kインデックス（y_trueの上位k個）
        topk_k = self.config.topk_k
        # argsortは昇順なので、末尾からk個を取って降順に
        topk_indices = np.argsort(y_true)[-topk_k:][::-1].astype(np.int64)

        # y_obs（観測値）: y_trueを[5, 11]でclip
        obs_low, obs_high = self.config.obs_clip_range
        y_obs = np.clip(y_true, obs_low, obs_high)

        return Matrix(
            y_true=y_true,
            y_obs=y_obs,
            main_effects=main_effects,
            global_bias=global_bias,
            slot_biases=slot_biases,
            errors=errors,
            top1_index=top1_index,
            topk_indices=topk_indices,
        )

    def _compute_y_true(
        self,
        global_bias: float,
        main_effects: List[NDArray[np.float64]],
        errors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        y_true配列を計算

        Args:
            global_bias: グローバルバイアス
            main_effects: 各スロットの主作用配列
            errors: 各セルの誤差配列

        Returns:
            y_true配列（長さ n_total）

        Note:
            効率的な計算のため、ブロードキャストを活用
        """
        n_total = self.indexer.n_total

        # 全セルの座標を取得
        all_indices = np.arange(n_total, dtype=np.int64)
        all_coords = self.indexer.batch_indices_to_coords(all_indices)

        # y_true = global_bias + Σ main_effects[slot][bb] + error
        y_true = np.full(n_total, global_bias, dtype=np.float64)

        for slot_idx, main_effect in enumerate(main_effects):
            bb_indices = all_coords[:, slot_idx]
            y_true += main_effect[bb_indices]

        y_true += errors

        return y_true
