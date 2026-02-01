"""
Random Operator for StageIII Simulator

完全ランダム戦略: 未開示セルから一様ランダムに選択
"""

from typing import List

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.RANDOM)
class RandomOperator(BaseOperator):
    """
    完全ランダム戦略

    未開示セルから一様ランダムにK個を選択する。
    学習やモデル更新は行わない。
    """

    name = "Random"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """
        未開示セルから一様ランダムにK個を選択

        Args:
            disclosure_state: 現在の開示状態
            k: 選択するセル数

        Returns:
            選択したセルのインデックスリスト
        """
        undisclosed = disclosure_state.get_undisclosed_indices()

        # 未開示セル数がk未満の場合は全て選択
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []

        selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
        return selected.tolist()

    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """ランダム戦略では更新不要"""
        pass

    def reset(self) -> None:
        """ランダム戦略ではリセット不要"""
        pass
