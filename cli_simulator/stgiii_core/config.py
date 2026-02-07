"""
Configuration classes for StageIII Simulator
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class OperatorType(Enum):
    """Operator戦略の種別"""
    RANDOM = "Random"
    FW_OLS = "Free-Wilson (OLS)"
    FW_RIDGE = "Free-Wilson (Ridge)"
    BAYESIAN_FW_UCB = "Bayesian Free-Wilson"
    BAYESIAN_FW_TS = "Bayesian Free-Wilson (TS)"
    FW_OLS_DISCRETE = "Free-Wilson (OLS + Discrete Interaction)"
    FW_RIDGE_DISCRETE = "Free-Wilson (Ridge + Discrete Interaction)"
    BAYESIAN_FW_DISCRETE = "Bayesian Free-Wilson (Discrete Interaction)"
    FW_OLS_CONTINUOUS = "Free-Wilson (OLS + Continuous Interaction)"
    FW_RIDGE_CONTINUOUS = "Free-Wilson (Ridge + Continuous Interaction)"
    BAYESIAN_FW_CONTINUOUS = "Bayesian Free-Wilson (Continuous Interaction)"


class NonlinearityType(Enum):
    """高次元変換の非線形種別"""
    TANH = "tanh"
    GELU = "gelu"


class ContinuousInteractionModel(Enum):
    """連続相互作用モデルの種別"""
    KRON = "kron"
    LOW_RANK = "low_rank"


class InitialDisclosureType(Enum):
    """初期開示の方式"""
    CROSS = "Cross"  # 従来方式：N-1スロット固定、1スロット全開示の和集合
    RANDOM_EACH_BB = "Random Each BB"  # 各BBが1回登場するようにランダム選択
    NONE = "None"  # 初期開示なし


@dataclass(frozen=True)
class SlotConfig:
    """スロット設定"""
    name: str                    # スロット名（"A", "B", "C", "D"）
    n_building_blocks: int       # BB数（10〜50）

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("スロット名は空にできません")
        if not (10 <= self.n_building_blocks <= 50):
            raise ValueError(
                f"BB数は10〜50である必要があります: {self.name}={self.n_building_blocks}"
            )


@dataclass(frozen=True)
class SimulationConfig:
    """シミュレーション全体の設定（イミュータブル）"""

    # ユーザ指定パラメータ
    operator_type: OperatorType
    n_trials: int                                    # 試行数（10〜1000）
    slots: Tuple[SlotConfig, ...]                    # スロット設定（2〜4個）
    k_per_step: int                                  # 1ステップで開示するセル数K（1〜10）
    topk_k: int = field(default=100)                 # Top-kのk値（固定: 100）
    initial_disclosure_type: InitialDisclosureType = InitialDisclosureType.NONE  # 初期開示方式（Noneのみ）
    random_seed: int | None = None                   # 再現性用シード（オプション）

    # Operator拡張（連続相互作用の入力・モデル設定）
    operator_high_dim: int = field(default=256)      # 高次元変換の出力次元
    operator_pca_dim: int = field(default=16)        # PCA次元（固定）
    operator_mlp_hidden_dim: int = field(default=64) # 高次元変換の中間次元
    operator_nonlinearity: NonlinearityType = field(default=NonlinearityType.TANH)
    continuous_interaction_model: ContinuousInteractionModel = field(
        default=ContinuousInteractionModel.KRON
    )
    continuous_interaction_rank: int = field(default=4)  # low-rank使用時のrank

    # v1.0 生成モデル（Main + Interaction(smooth+spike) + Residual）
    # UIでは主に「寄与比率」「cliff」「残差heavy-tail」を動かす想定
    f_main: float = field(default=0.3)               # main寄与（分散比率、合計=1）
    f_int: float = field(default=0.3)                # interaction寄与（分散比率）
    f_res: float = field(default=0.4)                # residual寄与（分散比率）

    # スロット距離（デフォルト: 直鎖距離 A-B-C-D）
    slot_distance_matrix: Tuple[Tuple[float, ...], ...] | None = None
    distance_lambda: float = field(default=1.0)      # scale=exp(-D/lambda)

    # BB埋め込み（シリーズ混合）
    embedding_dim: int = field(default=16)           # d
    embedding_series_divisor: int = field(default=5) # K_s=ceil(N_s/divisor)
    embedding_sigma: float = field(default=0.3)      # σ_z

    # 相互作用（smooth）
    interaction_rank: int = field(default=4)         # r（低ランク）

    # 相互作用（spike / activity cliff）
    eta_spike: float = field(default=0.3)            # cliff寄与（0=smoothのみ, 1=cliff支配）
    spike_hotspots: int = field(default=2)           # H（ホットスポット数、各スロット対で使用）
    spike_nu: float = field(default=6.0)             # t分布の自由度（振幅のheavy-tail）
    spike_sigma: float = field(default=1.0)          # 振幅スケール
    spike_ell: float = field(default=0.4)            # ホットスポット幅（UI必須ではない）

    # 残差（heavy-tail）
    residual_nu: float = field(default=6.0)          # t分布の自由度（外れやすさ）

    # 出力レンジ（obs_clip_range）は現状維持。生成後にσ_targetを調整してclip率を制御可能。
    clip_rate_max: float = field(default=0.01)       # 生成後の許容clip率（0〜1）

    # システム内部デフォルト（変更不可）
    global_bias: float = field(default=8.0)  # 出力平均の目安（obsレンジ中央を推奨）
    ridge_alpha: float = field(default=1.0)
    alpha_min: float = field(default=1e-6)
    alpha_max: float = field(default=1e6)
    sigma_min: float = field(default=0.05)
    sigma_iter_max: int = field(default=5)
    sigma_convergence_threshold: float = field(default=1e-3)
    ucb_beta: float = field(default=1.0)
    obs_clip_range: Tuple[float, float] = field(default=(5.0, 11.0))
    max_matrix_regeneration: int = field(default=5)
    max_initial_bb_retry: int = field(default=100)
    max_total_cells: int = field(default=20_000)

    def __post_init__(self) -> None:
        """バリデーション"""
        # スロット数チェック
        if not (2 <= len(self.slots) <= 4):
            raise ValueError(
                f"スロット数は2〜4である必要があります: {len(self.slots)}"
            )

        # 試行数チェック
        if not (1 <= self.n_trials <= 10000):
            raise ValueError(
                f"試行数は1〜10000である必要があります: {self.n_trials}"
            )

        # K値チェック
        if not (1 <= self.k_per_step <= 10):
            raise ValueError(
                f"1ステップの開示数Kは1〜10である必要があります: {self.k_per_step}"
            )

        # Top-k 固定（100）
        if self.topk_k != 100:
            raise ValueError(
                f"Top-kのkは100で固定です: {self.topk_k}"
            )

        # Operator拡張設定
        if self.operator_high_dim not in (256, 512):
            raise ValueError(
                f"operator_high_dimは256または512である必要があります: {self.operator_high_dim}"
            )
        if self.operator_pca_dim not in (4, 8, 16, 32, 64):
            raise ValueError(
                f"operator_pca_dimは4/8/16/32/64のいずれかである必要があります: {self.operator_pca_dim}"
            )
        if self.operator_mlp_hidden_dim not in (4, 8, 16, 32, 64):
            raise ValueError(
                "operator_mlp_hidden_dimは4/8/16/32/64のいずれかである必要があります: "
                f"{self.operator_mlp_hidden_dim}"
            )
        if self.operator_pca_dim > self.operator_high_dim:
            raise ValueError(
                "operator_pca_dimはoperator_high_dim以下である必要があります: "
                f"{self.operator_pca_dim} > {self.operator_high_dim}"
            )
        if self.continuous_interaction_model == ContinuousInteractionModel.LOW_RANK:
            if not (1 <= self.continuous_interaction_rank <= self.operator_pca_dim):
                raise ValueError(
                    "continuous_interaction_rankは1〜operator_pca_dimである必要があります: "
                    f"{self.continuous_interaction_rank}"
                )

        # 寄与比率チェック（分散比率、合計=1）
        for name, v in (("f_main", self.f_main), ("f_int", self.f_int), ("f_res", self.f_res)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}は0〜1である必要があります: {v}")

        f_sum = self.f_main + self.f_int + self.f_res
        if abs(f_sum - 1.0) > 1e-6:
            raise ValueError(
                f"f_main+f_int+f_resは1.0である必要があります: {f_sum}"
            )

        # スロット距離スケール
        if self.distance_lambda <= 0:
            raise ValueError(f"distance_lambdaは正である必要があります: {self.distance_lambda}")

        # 埋め込み
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dimは正である必要があります: {self.embedding_dim}")
        if self.embedding_series_divisor <= 0:
            raise ValueError(
                f"embedding_series_divisorは正である必要があります: {self.embedding_series_divisor}"
            )
        if self.embedding_sigma <= 0:
            raise ValueError(f"embedding_sigmaは正である必要があります: {self.embedding_sigma}")

        # 相互作用（smooth）
        if not (1 <= self.interaction_rank <= self.embedding_dim):
            raise ValueError(
                f"interaction_rankは1〜embedding_dimである必要があります: {self.interaction_rank}"
            )

        # 相互作用（spike）
        if not (0.0 <= self.eta_spike <= 1.0):
            raise ValueError(f"eta_spikeは0〜1である必要があります: {self.eta_spike}")
        if self.spike_hotspots < 0:
            raise ValueError(f"spike_hotspotsは0以上である必要があります: {self.spike_hotspots}")
        if self.spike_nu <= 2.0:
            raise ValueError(f"spike_nuは2より大きい必要があります: {self.spike_nu}")
        if self.spike_sigma <= 0:
            raise ValueError(f"spike_sigmaは正である必要があります: {self.spike_sigma}")
        if self.spike_ell <= 0:
            raise ValueError(f"spike_ellは正である必要があります: {self.spike_ell}")

        # 残差
        if self.residual_nu <= 2.0:
            raise ValueError(f"residual_nuは2より大きい必要があります: {self.residual_nu}")

        # clip率
        if not (0.0 <= self.clip_rate_max <= 1.0):
            raise ValueError(f"clip_rate_maxは0〜1である必要があります: {self.clip_rate_max}")

        # 初期開示方式（Noneのみ）
        if self.initial_disclosure_type != InitialDisclosureType.NONE:
            raise ValueError(
                f"初期開示方式はNoneのみ許可されています: {self.initial_disclosure_type}"
            )

        # Bayesian alpha bounds
        if self.ridge_alpha <= 0:
            raise ValueError(f"ridge_alphaは正である必要があります: {self.ridge_alpha}")
        if self.alpha_min <= 0 or self.alpha_max <= 0:
            raise ValueError(
                f"alpha_min/alpha_maxは正である必要があります: {self.alpha_min}, {self.alpha_max}"
            )
        if self.alpha_min >= self.alpha_max:
            raise ValueError(
                f"alpha_minはalpha_maxより小さい必要があります: {self.alpha_min} >= {self.alpha_max}"
            )

        # スロット距離行列（Advanced指定）
        if self.slot_distance_matrix is not None:
            if len(self.slot_distance_matrix) != self.n_slots:
                raise ValueError(
                    "slot_distance_matrixの次元がスロット数と一致しません"
                )
            for row in self.slot_distance_matrix:
                if len(row) != self.n_slots:
                    raise ValueError(
                        "slot_distance_matrixは正方行列である必要があります"
                    )
            for i in range(self.n_slots):
                if self.slot_distance_matrix[i][i] != 0:
                    raise ValueError(
                        "slot_distance_matrixの対角成分は0である必要があります"
                    )
                for j in range(self.n_slots):
                    if self.slot_distance_matrix[i][j] < 0:
                        raise ValueError(
                            "slot_distance_matrixは非負である必要があります"
                        )
                    # 対称性は推奨（必須ではないが、現実的には対称）
                    # if self.slot_distance_matrix[i][j] != self.slot_distance_matrix[j][i]:
                    #     raise ValueError("slot_distance_matrixは対称である必要があります")

        # 総セル数チェック
        if self.n_total_cells > self.max_total_cells:
            raise ValueError(
                f"総セル数が上限を超えています: {self.n_total_cells:,} > {self.max_total_cells:,}"
            )

    @property
    def n_slots(self) -> int:
        """スロット数"""
        return len(self.slots)

    @property
    def n_total_cells(self) -> int:
        """総セル数"""
        result = 1
        for slot in self.slots:
            result *= slot.n_building_blocks
        return result

    @property
    def slot_sizes(self) -> Tuple[int, ...]:
        """各スロットのBB数のタプル"""
        return tuple(s.n_building_blocks for s in self.slots)

    @property
    def slot_names(self) -> Tuple[str, ...]:
        """各スロットの名前のタプル"""
        return tuple(s.name for s in self.slots)

    @property
    def sigma_gen(self) -> float:
        """初期σの目安（obsレンジから導く、Bayesian-FWの初期値用途）"""
        low, high = self.obs_clip_range
        return (high - low) / 6.0

    def get_slot_distance_matrix(self) -> Tuple[Tuple[float, ...], ...]:
        """
        スロット距離行列Dを取得（未指定なら直鎖距離）

        Returns:
            D（shape: (n_slots, n_slots)）をタプルのタプルで返す
        """
        if self.slot_distance_matrix is not None:
            return self.slot_distance_matrix

        # 直鎖距離: D[i,j] = |i-j|
        n = self.n_slots
        return tuple(
            tuple(float(abs(i - j)) for j in range(n))
            for i in range(n)
        )

    def to_dict(self) -> dict:
        """設定を辞書形式に変換"""
        return {
            "operator_type": self.operator_type.value,
            "n_trials": self.n_trials,
            "n_slots": self.n_slots,
            "slot_sizes": self.slot_sizes,
            "slot_names": self.slot_names,
            "n_total_cells": self.n_total_cells,
            "operator": {
                "high_dim": self.operator_high_dim,
                "pca_dim": self.operator_pca_dim,
                "mlp_hidden_dim": self.operator_mlp_hidden_dim,
                "nonlinearity": self.operator_nonlinearity.value,
                "continuous_model": self.continuous_interaction_model.value,
                "continuous_rank": self.continuous_interaction_rank,
            },
            "generation": {
                "f_main": self.f_main,
                "f_int": self.f_int,
                "f_res": self.f_res,
                "distance_lambda": self.distance_lambda,
                "embedding_dim": self.embedding_dim,
                "embedding_series_divisor": self.embedding_series_divisor,
                "embedding_sigma": self.embedding_sigma,
                "interaction_rank": self.interaction_rank,
                "eta_spike": self.eta_spike,
                "spike_hotspots": self.spike_hotspots,
                "spike_nu": self.spike_nu,
                "spike_sigma": self.spike_sigma,
                "spike_ell": self.spike_ell,
                "residual_nu": self.residual_nu,
                "clip_rate_max": self.clip_rate_max,
            },
            "k_per_step": self.k_per_step,
            "topk_k": self.topk_k,
            "initial_disclosure_type": self.initial_disclosure_type.value,
            "random_seed": self.random_seed,
        }
