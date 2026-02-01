"""
Cell index converter for StageIII Simulator

線形インデックスとN次元座標の相互変換を提供
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class CellIndexer:
    """線形インデックスとN次元座標の相互変換"""

    slot_sizes: Tuple[int, ...]  # 各スロットのBB数

    # 内部計算結果（post_initで初期化）
    _strides: Tuple[int, ...] = field(init=False, repr=False)
    _n_total: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """ストライドと総セル数を計算"""
        self._strides = self._compute_strides()
        self._n_total = int(np.prod(self.slot_sizes))

    def _compute_strides(self) -> Tuple[int, ...]:
        """
        各スロットのストライドを計算（row-major order）

        例: slot_sizes = (3, 4, 5) の場合
            strides = (20, 5, 1)
            index = i * 20 + j * 5 + k * 1
        """
        strides = []
        stride = 1
        for size in reversed(self.slot_sizes):
            strides.append(stride)
            stride *= size
        return tuple(reversed(strides))

    @property
    def n_total(self) -> int:
        """総セル数"""
        return self._n_total

    @property
    def n_slots(self) -> int:
        """スロット数"""
        return len(self.slot_sizes)

    def coords_to_index(self, coords: Tuple[int, ...]) -> int:
        """
        N次元座標を線形インデックスに変換

        Args:
            coords: 各スロットのBBインデックス（0-indexed）
                   例: (2, 1, 3) は A=2番目, B=1番目, C=3番目のBB

        Returns:
            線形インデックス（0 <= index < n_total）

        Raises:
            ValueError: 座標の次元がスロット数と一致しない場合
            ValueError: 座標が各スロットの範囲外の場合
        """
        if len(coords) != len(self.slot_sizes):
            raise ValueError(
                f"座標の次元が不正: {len(coords)} != {len(self.slot_sizes)}"
            )

        index = 0
        for i, (coord, stride, size) in enumerate(
            zip(coords, self._strides, self.slot_sizes)
        ):
            if not (0 <= coord < size):
                raise ValueError(
                    f"座標が範囲外: slot[{i}]={coord}, 有効範囲=[0, {size})"
                )
            index += coord * stride

        return index

    def index_to_coords(self, index: int) -> Tuple[int, ...]:
        """
        線形インデックスをN次元座標に変換

        Args:
            index: 線形インデックス（0 <= index < n_total）

        Returns:
            各スロットのBBインデックス（0-indexed）

        Raises:
            ValueError: インデックスが範囲外の場合
        """
        if not (0 <= index < self._n_total):
            raise ValueError(
                f"インデックスが範囲外: {index}, 有効範囲=[0, {self._n_total})"
            )

        coords = []
        remaining = index
        for stride in self._strides:
            coords.append(remaining // stride)
            remaining %= stride

        return tuple(coords)

    def batch_coords_to_indices(
        self, coords_array: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """
        複数の座標を一括で線形インデックスに変換

        Args:
            coords_array: 座標配列（shape: (n_cells, n_slots)）

        Returns:
            線形インデックス配列（shape: (n_cells,)）

        Note:
            バリデーションは行わない（高速化のため）
        """
        strides = np.array(self._strides, dtype=np.int64)
        return np.sum(coords_array * strides, axis=1).astype(np.int64)

    def batch_indices_to_coords(
        self, indices: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """
        複数の線形インデックスを一括で座標に変換

        Args:
            indices: 線形インデックス配列（shape: (n_cells,)）

        Returns:
            座標配列（shape: (n_cells, n_slots)）

        Note:
            バリデーションは行わない（高速化のため）
        """
        n_cells = len(indices)
        coords = np.zeros((n_cells, len(self.slot_sizes)), dtype=np.int64)
        remaining = indices.copy()

        for i, stride in enumerate(self._strides):
            coords[:, i] = remaining // stride
            remaining %= stride

        return coords

    def get_all_indices(self) -> NDArray[np.int64]:
        """全セルのインデックス配列を取得"""
        return np.arange(self._n_total, dtype=np.int64)

    def get_slot_indices_for_fixed(
        self,
        fixed_coords: Tuple[int, ...],
        vary_slot: int
    ) -> NDArray[np.int64]:
        """
        特定のスロット以外を固定したときのセルインデックスを取得

        Args:
            fixed_coords: 固定する座標（vary_slotの値は無視される）
            vary_slot: 変化させるスロットのインデックス

        Returns:
            セルインデックス配列（長さ = slot_sizes[vary_slot]）
        """
        n_bb = self.slot_sizes[vary_slot]
        indices = []

        for bb_idx in range(n_bb):
            coords = list(fixed_coords)
            coords[vary_slot] = bb_idx
            indices.append(self.coords_to_index(tuple(coords)))

        return np.array(indices, dtype=np.int64)
