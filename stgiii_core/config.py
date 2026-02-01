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
    main_effect_range: Tuple[float, float]           # 主作用の一様分布範囲 [low, high]
    error_clip_range: Tuple[float, float]            # 誤差のclip範囲 [low, high]
    k_per_step: int                                  # 1ステップで開示するセル数K（1〜10）
    topk_k: int                                      # Top-kのk値（5, 10, 25, 50, 100）
    initial_disclosure_type: InitialDisclosureType = InitialDisclosureType.CROSS  # 初期開示方式
    random_seed: int | None = None                   # 再現性用シード（オプション）

    # システム内部デフォルト（変更不可）
    global_bias: float = field(default=6.0)  # 固定グローバルバイアス
    slot_bias_range: Tuple[float, float] = field(default=(-0.5, 0.5))
    ridge_alpha: float = field(default=1.0)
    sigma_min: float = field(default=0.05)
    sigma_iter_max: int = field(default=5)
    sigma_convergence_threshold: float = field(default=1e-3)
    ucb_beta: float = field(default=1.0)
    obs_clip_range: Tuple[float, float] = field(default=(5.0, 11.0))
    max_matrix_regeneration: int = field(default=5)
    max_initial_bb_retry: int = field(default=100)
    max_total_cells: int = field(default=100_000)

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

        # Top-k チェック
        if self.topk_k not in (5, 10, 25, 50, 100):
            raise ValueError(
                f"Top-kのkは5, 10, 25, 50, 100のいずれかである必要があります: {self.topk_k}"
            )

        # 主作用範囲チェック
        if self.main_effect_range[0] >= self.main_effect_range[1]:
            raise ValueError(
                f"主作用範囲の下限は上限より小さい必要があります: {self.main_effect_range}"
            )

        # 誤差範囲チェック
        if self.error_clip_range[0] >= self.error_clip_range[1]:
            raise ValueError(
                f"誤差範囲の下限は上限より小さい必要があります: {self.error_clip_range}"
            )

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
        """データ生成用の誤差標準偏差（±3σが概ね範囲に収まる想定）"""
        low, high = self.error_clip_range
        return (high - low) / 6.0

    def to_dict(self) -> dict:
        """設定を辞書形式に変換"""
        return {
            "operator_type": self.operator_type.value,
            "n_trials": self.n_trials,
            "n_slots": self.n_slots,
            "slot_sizes": self.slot_sizes,
            "slot_names": self.slot_names,
            "n_total_cells": self.n_total_cells,
            "main_effect_range": self.main_effect_range,
            "error_clip_range": self.error_clip_range,
            "k_per_step": self.k_per_step,
            "topk_k": self.topk_k,
            "initial_disclosure_type": self.initial_disclosure_type.value,
            "random_seed": self.random_seed,
        }
