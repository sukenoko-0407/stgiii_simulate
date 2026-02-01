"""
Tests for CellIndexer
"""

import pytest
import numpy as np

from stgiii_core.indexer import CellIndexer


class TestCellIndexer:
    """CellIndexerの単体テスト"""

    def test_init_2slots(self) -> None:
        """2スロットでの初期化"""
        indexer = CellIndexer((3, 4))
        assert indexer.n_total == 12
        assert indexer.n_slots == 2

    def test_init_3slots(self) -> None:
        """3スロットでの初期化"""
        indexer = CellIndexer((3, 4, 5))
        assert indexer.n_total == 60
        assert indexer.n_slots == 3

    def test_init_4slots(self) -> None:
        """4スロットでの初期化"""
        indexer = CellIndexer((2, 3, 4, 5))
        assert indexer.n_total == 120
        assert indexer.n_slots == 4

    def test_coords_to_index_2slots(self) -> None:
        """2スロットでの座標→インデックス変換"""
        indexer = CellIndexer((3, 4))
        assert indexer.coords_to_index((0, 0)) == 0
        assert indexer.coords_to_index((0, 1)) == 1
        assert indexer.coords_to_index((0, 3)) == 3
        assert indexer.coords_to_index((1, 0)) == 4
        assert indexer.coords_to_index((2, 3)) == 11

    def test_coords_to_index_3slots(self) -> None:
        """3スロットでの座標→インデックス変換"""
        indexer = CellIndexer((2, 3, 4))
        assert indexer.coords_to_index((0, 0, 0)) == 0
        assert indexer.coords_to_index((0, 0, 1)) == 1
        assert indexer.coords_to_index((0, 1, 0)) == 4
        assert indexer.coords_to_index((1, 0, 0)) == 12
        assert indexer.coords_to_index((1, 2, 3)) == 23

    def test_index_to_coords_2slots(self) -> None:
        """2スロットでのインデックス→座標変換"""
        indexer = CellIndexer((3, 4))
        assert indexer.index_to_coords(0) == (0, 0)
        assert indexer.index_to_coords(1) == (0, 1)
        assert indexer.index_to_coords(4) == (1, 0)
        assert indexer.index_to_coords(11) == (2, 3)

    def test_index_to_coords_3slots(self) -> None:
        """3スロットでのインデックス→座標変換"""
        indexer = CellIndexer((2, 3, 4))
        assert indexer.index_to_coords(0) == (0, 0, 0)
        assert indexer.index_to_coords(1) == (0, 0, 1)
        assert indexer.index_to_coords(4) == (0, 1, 0)
        assert indexer.index_to_coords(12) == (1, 0, 0)
        assert indexer.index_to_coords(23) == (1, 2, 3)

    def test_roundtrip(self) -> None:
        """座標⇔インデックスの往復変換"""
        indexer = CellIndexer((10, 20, 15))
        for idx in range(indexer.n_total):
            coords = indexer.index_to_coords(idx)
            assert indexer.coords_to_index(coords) == idx

    def test_batch_coords_to_indices(self) -> None:
        """バッチ座標→インデックス変換"""
        indexer = CellIndexer((5, 6, 7))
        coords = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
        ], dtype=np.int64)
        indices = indexer.batch_coords_to_indices(coords)
        assert len(indices) == 3
        assert indices[0] == indexer.coords_to_index((0, 0, 0))
        assert indices[1] == indexer.coords_to_index((1, 2, 3))
        assert indices[2] == indexer.coords_to_index((4, 5, 6))

    def test_batch_indices_to_coords(self) -> None:
        """バッチインデックス→座標変換"""
        indexer = CellIndexer((5, 6, 7))
        indices = np.array([0, 10, 50, 100], dtype=np.int64)
        coords = indexer.batch_indices_to_coords(indices)
        assert coords.shape == (4, 3)
        for i, idx in enumerate(indices):
            expected = indexer.index_to_coords(int(idx))
            assert tuple(coords[i]) == expected

    def test_batch_roundtrip(self) -> None:
        """バッチ操作の往復変換"""
        indexer = CellIndexer((5, 6, 7))
        indices = np.array([0, 10, 50, 100], dtype=np.int64)
        coords = indexer.batch_indices_to_coords(indices)
        recovered = indexer.batch_coords_to_indices(coords)
        np.testing.assert_array_equal(indices, recovered)

    def test_coords_dimension_mismatch(self) -> None:
        """座標次元の不一致でエラー"""
        indexer = CellIndexer((3, 4, 5))
        with pytest.raises(ValueError, match="座標の次元が不正"):
            indexer.coords_to_index((1, 2))  # 2要素だが3要素必要

    def test_coords_out_of_range(self) -> None:
        """座標範囲外でエラー"""
        indexer = CellIndexer((3, 4, 5))
        with pytest.raises(ValueError, match="座標が範囲外"):
            indexer.coords_to_index((3, 0, 0))  # slot 0は0-2まで

    def test_index_out_of_range(self) -> None:
        """インデックス範囲外でエラー"""
        indexer = CellIndexer((3, 4, 5))
        with pytest.raises(ValueError, match="インデックスが範囲外"):
            indexer.index_to_coords(100)  # 0-59まで有効

    def test_get_slot_indices_for_fixed(self) -> None:
        """固定座標での1スロット全開示"""
        indexer = CellIndexer((3, 4, 5))
        indices = indexer.get_slot_indices_for_fixed((1, 2, 3), vary_slot=0)
        assert len(indices) == 3  # slot 0のサイズ
        # slot 0を変化させて他は固定
        for bb_idx, idx in enumerate(indices):
            coords = indexer.index_to_coords(int(idx))
            assert coords[0] == bb_idx
            assert coords[1] == 2
            assert coords[2] == 3
