"""
Base class for Operators in StageIII Simulator

探索戦略（Operator）の抽象基底クラスとコンテキスト定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Optional
import numpy as np
from numpy.typing import NDArray

from ..config import SimulationConfig
from ..indexer import CellIndexer
from ..disclosure import DisclosureState
from ..matrix import OperatorFeatureBundle


class OperatorProtocol(Protocol):
    """Operatorが満たすべきインターフェース（静的型チェック用）"""

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """次に開示するセルを選択"""
        ...

    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """新規開示データでモデルを更新"""
        ...


@dataclass
class OperatorContext:
    """Operatorに渡すコンテキスト情報"""

    config: SimulationConfig
    indexer: CellIndexer
    rng: np.random.Generator
    features: Optional[OperatorFeatureBundle] = None


class BaseOperator(ABC):
    """
    Operator抽象基底クラス

    全ての探索戦略はこのクラスを継承して実装する。
    """

    # サブクラスで定義必須
    name: str = ""

    def __init__(self, context: OperatorContext) -> None:
        """
        Args:
            context: Operatorコンテキスト（設定、インデクサ、乱数生成器）
        """
        self.context = context
        self.config = context.config
        self.indexer = context.indexer
        self.rng = context.rng

    @abstractmethod
    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """
        次に開示するセルを選択

        Args:
            disclosure_state: 現在の開示状態
            k: 選択するセル数

        Returns:
            選択したセルのインデックスリスト（長さ k）

        Note:
            - 未開示セルからのみ選択すること
            - 既開示セルを選択してはならない
            - 未開示セル数がk未満の場合は残り全てを選択
        """
        pass

    @abstractmethod
    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """
        新規開示データでモデルを更新

        Args:
            new_indices: 新規開示セルのインデックスリスト
            new_values: 対応する観測値リスト

        Note:
            学習が不要な戦略（Randomなど）ではpassでよい
        """
        pass

    def reset(self) -> None:
        """
        Operatorの内部状態をリセット（新規試行開始時に呼ばれる）

        Note:
            サブクラスで必要に応じてオーバーライド
        """
        pass

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64]
    ) -> None:
        """
        開示済み全データでモデルを学習

        Args:
            indices: 開示済みセルのインデックス配列
            values: 対応する観測値配列

        Note:
            デフォルトではupdate()を呼ぶだけ。
            サブクラスで必要に応じてオーバーライド。
        """
        self.update(indices.tolist(), values.tolist())

    def _validate_selection(
        self,
        selected: List[int],
        disclosure_state: DisclosureState
    ) -> None:
        """
        選択の妥当性を検証

        Args:
            selected: 選択されたセルインデックスリスト
            disclosure_state: 開示状態

        Raises:
            ValueError: 既開示セルが選択されている場合
        """
        for idx in selected:
            if disclosure_state.is_disclosed(idx):
                raise ValueError(f"既開示セルを選択しました: {idx}")

    def _random_tiebreak(
        self,
        candidates: NDArray[np.int64],
        scores: NDArray[np.float64],
        k: int
    ) -> List[int]:
        """
        スコア上位k個を選択（同点時はランダム）

        Args:
            candidates: 候補セルのインデックス配列
            scores: 対応するスコア配列
            k: 選択数

        Returns:
            選択されたインデックスのリスト
        """
        n = len(candidates)
        if n == 0:
            return []
        if n <= k:
            return candidates.tolist()

        # スコアでソート（降順）、同点時はランダム順
        random_tiebreaker = self.rng.random(n)
        # lexsortは末尾のキーが優先されるので、(-scores, random)の順
        sorted_idx = np.lexsort((random_tiebreaker, -scores))
        return candidates[sorted_idx[:k]].tolist()

    def _get_feature_count(self) -> int:
        """
        Reference codingでの特徴量数を計算

        Returns:
            特徴量数 = 1 (intercept) + sum(n_bb - 1 for each slot)
        """
        n_features = 1  # intercept
        for size in self.config.slot_sizes:
            n_features += size - 1
        return n_features

    def _build_design_matrix(
        self,
        indices: NDArray[np.int64] | List[int]
    ) -> NDArray[np.float64]:
        """
        Reference codingによる設計行列を構築

        Args:
            indices: セルインデックス配列

        Returns:
            設計行列 X（shape: (n_samples, n_features)）

        Note:
            各スロットの最初のBB（インデックス0）を基準カテゴリとする
            特徴量: [intercept, A_1, A_2, ..., B_1, B_2, ..., ...]
        """
        indices_arr = np.asarray(indices, dtype=np.int64)
        coords = self.indexer.batch_indices_to_coords(indices_arr)
        n_samples = len(indices_arr)

        n_features = self._get_feature_count()
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        X[:, 0] = 1.0  # intercept

        col_offset = 1
        for slot_idx, size in enumerate(self.config.slot_sizes):
            for bb_idx in range(1, size):  # 0番目は基準なのでスキップ
                mask = coords[:, slot_idx] == bb_idx
                X[mask, col_offset] = 1.0
                col_offset += 1

        return X
