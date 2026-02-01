"""
Disclosure state management for StageIII Simulator

開示状態（どのセルが開示済みか、その観測値）を管理
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy.typing import NDArray


@dataclass
class DisclosureState:
    """開示状態を管理するクラス"""

    n_total: int  # 総セル数

    # 内部状態（post_initで初期化）
    _disclosed_mask: NDArray[np.bool_] = field(init=False, repr=False)
    _disclosed_indices: List[int] = field(default_factory=list, repr=False)
    _disclosed_values: List[float] = field(default_factory=list, repr=False)
    _disclosure_order: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """初期化"""
        self._disclosed_mask = np.zeros(self.n_total, dtype=bool)
        # default_factoryで初期化されるが、frozen=Falseなので再初期化
        self._disclosed_indices = []
        self._disclosed_values = []
        self._disclosure_order = []

    @property
    def n_disclosed(self) -> int:
        """開示済みセル数（ユニーク）"""
        return int(np.sum(self._disclosed_mask))

    @property
    def disclosed_indices(self) -> NDArray[np.int64]:
        """開示済みセルのインデックス配列（開示順）"""
        return np.array(self._disclosed_indices, dtype=np.int64)

    @property
    def disclosed_values(self) -> NDArray[np.float64]:
        """開示済みセルの観測値配列（開示順）"""
        return np.array(self._disclosed_values, dtype=np.float64)

    @property
    def disclosed_mask(self) -> NDArray[np.bool_]:
        """開示済みマスク（読み取り専用コピー）"""
        return self._disclosed_mask.copy()

    def is_disclosed(self, index: int) -> bool:
        """
        指定セルが開示済みか判定

        Args:
            index: セルインデックス

        Returns:
            開示済みならTrue
        """
        return bool(self._disclosed_mask[index])

    def disclose(
        self,
        indices: List[int],
        values: List[float]
    ) -> int:
        """
        セルを開示する

        Args:
            indices: 開示するセルのインデックスリスト
            values: 対応する観測値リスト

        Returns:
            新規に開示されたセル数（重複除外）

        Note:
            既開示セルを再度渡した場合は無視される（重複カウントなし）
        """
        if len(indices) != len(values):
            raise ValueError(
                f"indicesとvaluesの長さが一致しません: {len(indices)} != {len(values)}"
            )

        new_count = 0
        for idx, val in zip(indices, values):
            if not self._disclosed_mask[idx]:
                self._disclosed_mask[idx] = True
                self._disclosed_indices.append(idx)
                self._disclosed_values.append(val)
                self._disclosure_order.append(idx)
                new_count += 1

        return new_count

    def get_undisclosed_indices(self) -> NDArray[np.int64]:
        """
        未開示セルのインデックス配列を取得

        Returns:
            未開示セルのインデックス配列
        """
        return np.where(~self._disclosed_mask)[0].astype(np.int64)

    def get_undisclosed_count(self) -> int:
        """未開示セル数を取得"""
        return self.n_total - self.n_disclosed

    def contains_any(
        self,
        indices: NDArray[np.int64] | List[int]
    ) -> bool:
        """
        指定したインデックスのいずれかが開示済みか判定

        Args:
            indices: チェックするインデックス配列またはリスト

        Returns:
            いずれかが開示済みならTrue
        """
        indices_arr = np.asarray(indices)
        return bool(np.any(self._disclosed_mask[indices_arr]))

    def contains_all(
        self,
        indices: NDArray[np.int64] | List[int]
    ) -> bool:
        """
        指定したインデックスが全て開示済みか判定

        Args:
            indices: チェックするインデックス配列またはリスト

        Returns:
            全て開示済みならTrue
        """
        indices_arr = np.asarray(indices)
        return bool(np.all(self._disclosed_mask[indices_arr]))

    def get_data_for_training(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        学習用のデータ（インデックスと観測値）を取得

        Returns:
            (indices, values) のタプル
        """
        return self.disclosed_indices, self.disclosed_values

    def reset(self) -> None:
        """開示状態をリセット"""
        self._disclosed_mask[:] = False
        self._disclosed_indices.clear()
        self._disclosed_values.clear()
        self._disclosure_order.clear()

    def copy(self) -> "DisclosureState":
        """開示状態のコピーを作成"""
        new_state = DisclosureState(n_total=self.n_total)
        new_state._disclosed_mask = self._disclosed_mask.copy()
        new_state._disclosed_indices = self._disclosed_indices.copy()
        new_state._disclosed_values = self._disclosed_values.copy()
        new_state._disclosure_order = self._disclosure_order.copy()
        return new_state
