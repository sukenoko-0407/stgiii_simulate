"""
Simulation engine for StageIII Simulator

シミュレーションの実行エンジン
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Set, Tuple
import numpy as np

from .config import SimulationConfig, InitialDisclosureType
from .matrix import Matrix, MatrixGenerator
from .indexer import CellIndexer
from .disclosure import DisclosureState
from .operators.base import BaseOperator, OperatorContext
from .operators.registry import get_operator
from .results import TrialResult, SimulationResults


@dataclass
class InitialDisclosureResult:
    """初期開示の結果"""
    disclosed_indices: List[int]
    center_coords: Tuple[int, ...]
    contains_top1: bool


class SimulationEngine:
    """シミュレーション実行エンジン"""

    def __init__(
        self,
        config: SimulationConfig,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        """
        Args:
            config: シミュレーション設定
            progress_callback: 進捗コールバック (current, total) -> None
        """
        self.config = config
        self.progress_callback = progress_callback

        # 乱数生成器
        self.rng = np.random.default_rng(config.random_seed)

        # インデクサー
        self.indexer = CellIndexer(config.slot_sizes)

    def run(self) -> SimulationResults:
        """
        全試行を実行

        Returns:
            シミュレーション結果
        """
        trials: List[TrialResult] = []

        for trial_id in range(self.config.n_trials):
            if self.progress_callback:
                self.progress_callback(trial_id, self.config.n_trials)

            result = self._run_single_trial(trial_id)
            trials.append(result)

        if self.progress_callback:
            self.progress_callback(self.config.n_trials, self.config.n_trials)

        config_summary = self.config.to_dict()

        return SimulationResults(trials=trials, config_summary=config_summary)

    def _run_single_trial(self, trial_id: int) -> TrialResult:
        """
        単一試行を実行

        Args:
            trial_id: 試行ID

        Returns:
            試行結果
        """
        start_time = time.perf_counter()

        # Matrix生成
        generator = MatrixGenerator(self.config, self.indexer, self.rng)
        matrix = generator.generate()

        # Operator初期化
        context = OperatorContext(
            config=self.config,
            indexer=self.indexer,
            rng=self.rng
        )
        operator = get_operator(self.config.operator_type, context)
        operator.reset()

        # 開示状態初期化
        disclosure = DisclosureState(n_total=self.indexer.n_total)

        # 初期開示
        initial_result = self._initial_disclosure(matrix, disclosure)
        n_initial = disclosure.n_disclosed

        # 初期開示でTop-1/Top-k到達チェック
        hit_top1_initial = matrix.top1_index in initial_result.disclosed_indices
        hit_topk_initial = any(
            idx in initial_result.disclosed_indices
            for idx in matrix.topk_indices
        )
        topk_set = set(matrix.topk_indices.tolist())
        topk_hit_set = set(
            idx for idx in initial_result.disclosed_indices if idx in topk_set
        )
        topk_hit_count = len(topk_hit_set)

        # P_top1, P_topkの初期値
        p_top1: int | None = n_initial if hit_top1_initial else None
        p_topk: int | None = n_initial if hit_topk_initial else None
        p_top100_50: int | None = n_initial if topk_hit_count >= 50 else None

        # Operatorの初期学習
        indices, values = disclosure.get_data_for_training()
        operator.fit(indices, values)

        # 反復ステップ
        n_steps = 0
        k = self.config.k_per_step

        while (p_top1 is None) or (p_top100_50 is None):
            n_steps += 1

            # 次に開示するセルを選択
            selected = operator.select_next_cells(disclosure, k)

            if len(selected) == 0:
                # 全セル開示済み（通常はここに到達しない）
                break

            # 開示
            selected_values = [float(matrix.y_obs[idx]) for idx in selected]
            disclosure.disclose(selected, selected_values)

            # Operatorの更新（全データで再学習）
            indices, values = disclosure.get_data_for_training()
            operator.fit(indices, values)

            # Top-k到達チェック（バッチ内のいずれかがTop-kなら到達）
            if p_topk is None:
                for idx in selected:
                    if matrix.is_topk(idx):
                        p_topk = disclosure.n_disclosed
                        break

            # Top-100のうち50個に到達したか
            if p_top100_50 is None:
                new_hits = 0
                for idx in selected:
                    if idx in topk_set and idx not in topk_hit_set:
                        topk_hit_set.add(idx)
                        new_hits += 1
                if new_hits > 0:
                    topk_hit_count += new_hits
                if topk_hit_count >= 50:
                    p_top100_50 = disclosure.n_disclosed

            # Top-1到達チェック
            if matrix.top1_index in selected:
                p_top1 = disclosure.n_disclosed

        # Top-kが未到達のまま終了した場合（Top-1がTop-kに含まれていた場合）
        if p_topk is None:
            p_topk = p_top1
        if p_top1 is None:
            p_top1 = disclosure.n_disclosed
        if p_top100_50 is None:
            p_top100_50 = disclosure.n_disclosed

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TrialResult(
            trial_id=trial_id,
            method=self.config.operator_type.value,
            n_total_cells=self.config.n_total_cells,
            n_initial_disclosed=n_initial,
            k_value=k,
            topk_k=self.config.topk_k,
            p_top1=p_top1 if p_top1 is not None else disclosure.n_disclosed,
            p_topk=p_topk if p_topk is not None else disclosure.n_disclosed,
            p_top100_50=p_top100_50 if p_top100_50 is not None else disclosure.n_disclosed,
            n_steps=n_steps,
            hit_in_initial_top1=hit_top1_initial,
            hit_in_initial_topk=hit_topk_initial,
            runtime_ms=elapsed_ms,
        )

    def _initial_disclosure(
        self,
        matrix: Matrix,
        disclosure: DisclosureState
    ) -> InitialDisclosureResult:
        """
        初期開示を実行

        Args:
            matrix: 評価値マトリックス
            disclosure: 開示状態

        Returns:
            初期開示結果

        Note:
            正解セルが含まれる場合は再抽選
        """
        if self.config.initial_disclosure_type == InitialDisclosureType.CROSS:
            return self._initial_disclosure_cross(matrix, disclosure)
        elif self.config.initial_disclosure_type == InitialDisclosureType.RANDOM_EACH_BB:
            return self._initial_disclosure_random_each_bb(matrix, disclosure)
        else:  # NONE
            return self._initial_disclosure_none()

    def _initial_disclosure_cross(
        self,
        matrix: Matrix,
        disclosure: DisclosureState
    ) -> InitialDisclosureResult:
        """
        Cross方式の初期開示

        「N-1個のスロットを固定、残り1スロットを全開示」の和集合を開示

        Note:
            正解セルが含まれる場合は中心座標を再抽選
        """
        for attempt in range(self.config.max_initial_bb_retry):
            # 中心座標をランダムに選択
            center_coords = tuple(
                int(self.rng.integers(0, size))
                for size in self.config.slot_sizes
            )

            # 初期開示セル集合を構築
            disclosed_set: Set[int] = set()

            for vary_slot in range(self.config.n_slots):
                # vary_slot以外を固定、vary_slotを全開示
                for bb_idx in range(self.config.slot_sizes[vary_slot]):
                    coords = list(center_coords)
                    coords[vary_slot] = bb_idx
                    idx = self.indexer.coords_to_index(tuple(coords))
                    disclosed_set.add(idx)

            disclosed_indices = list(disclosed_set)

            # 正解セルが含まれていないかチェック
            if matrix.top1_index not in disclosed_set:
                # 開示を実行
                disclosed_values = [
                    float(matrix.y_obs[idx]) for idx in disclosed_indices
                ]
                disclosure.disclose(disclosed_indices, disclosed_values)

                return InitialDisclosureResult(
                    disclosed_indices=disclosed_indices,
                    center_coords=center_coords,
                    contains_top1=False,
                )

        # 正解を回避できなかった場合はそのまま続行
        # （極めて稀なケース、またはテスト用）
        disclosed_values = [
            float(matrix.y_obs[idx]) for idx in disclosed_indices
        ]
        disclosure.disclose(disclosed_indices, disclosed_values)

        return InitialDisclosureResult(
            disclosed_indices=disclosed_indices,
            center_coords=center_coords,
            contains_top1=True,
        )

    def _initial_disclosure_random_each_bb(
        self,
        matrix: Matrix,
        disclosure: DisclosureState
    ) -> InitialDisclosureResult:
        """
        Random Each BB方式の初期開示

        各スロットの各BBが少なくとも1回登場するようにランダム選択
        未開示BBの中からランダムに抽選する形式

        Note:
            別スロットにて2回以上登場するBBが存在することは許容
            正解セルが含まれる場合は再抽選
        """
        for attempt in range(self.config.max_initial_bb_retry):
            disclosed_set: Set[int] = set()

            # 各スロットについて、そのスロットのBBごとに1つのセルを選択
            for slot_idx in range(self.config.n_slots):
                n_bb = self.config.slot_sizes[slot_idx]

                for bb_idx in range(n_bb):
                    # 該当BBの座標を固定し、他のスロットはランダムに選択
                    coords = []
                    for other_slot in range(self.config.n_slots):
                        if other_slot == slot_idx:
                            coords.append(bb_idx)
                        else:
                            coords.append(
                                int(self.rng.integers(0, self.config.slot_sizes[other_slot]))
                            )

                    idx = self.indexer.coords_to_index(tuple(coords))
                    disclosed_set.add(idx)

            disclosed_indices = list(disclosed_set)

            # 正解セルが含まれていないかチェック
            if matrix.top1_index not in disclosed_set:
                # 開示を実行
                disclosed_values = [
                    float(matrix.y_obs[idx]) for idx in disclosed_indices
                ]
                disclosure.disclose(disclosed_indices, disclosed_values)

                return InitialDisclosureResult(
                    disclosed_indices=disclosed_indices,
                    center_coords=(),  # この方式では中心座標は使用しない
                    contains_top1=False,
                )

        # 正解を回避できなかった場合はそのまま続行
        disclosed_values = [
            float(matrix.y_obs[idx]) for idx in disclosed_indices
        ]
        disclosure.disclose(disclosed_indices, disclosed_values)

        return InitialDisclosureResult(
            disclosed_indices=disclosed_indices,
            center_coords=(),
            contains_top1=True,
        )

    def _initial_disclosure_none(self) -> InitialDisclosureResult:
        """
        初期開示なし

        何も開示せずに探索を開始する
        """
        return InitialDisclosureResult(
            disclosed_indices=[],
            center_coords=(),
            contains_top1=False,
        )

    @staticmethod
    def calculate_initial_disclosure_count(
        slot_sizes: Tuple[int, ...],
        disclosure_type: InitialDisclosureType = InitialDisclosureType.NONE
    ) -> int:
        """
        初期開示セル数を計算（静的メソッド、UI表示用）

        Args:
            slot_sizes: 各スロットのBB数
            disclosure_type: 初期開示方式

        Returns:
            初期開示セル数（ユニーク）

        Note:
            現仕様ではNoneのみ使用
        """
        if disclosure_type == InitialDisclosureType.CROSS:
            n_slots = len(slot_sizes)
            total = sum(slot_sizes)
            # 中心セルは各スロットの全開示で1回ずつ含まれる（計n_slots回）
            # 実際は1つなので、重複分 = n_slots - 1 を引く
            return total - (n_slots - 1)
        elif disclosure_type == InitialDisclosureType.RANDOM_EACH_BB:
            # 各BBが1回登場するので、合計BB数が理論上限
            # 重複により実際は少し減る可能性があるが、表示用には最大値を返す
            return sum(slot_sizes)
        else:  # NONE
            return 0

    @staticmethod
    def validate_config(config: SimulationConfig) -> List[str]:
        """
        設定のバリデーションを行い、警告メッセージを返す

        Args:
            config: シミュレーション設定

        Returns:
            警告メッセージのリスト（問題なければ空）
        """
        warnings: List[str] = []

        # 総セル数チェック
        if config.n_total_cells > config.max_total_cells:
            warnings.append(
                f"総セル数が上限を超えています: {config.n_total_cells:,} > {config.max_total_cells:,}"
            )

        # 初期開示はNone固定のため警告対象なし

        return warnings
