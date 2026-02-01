"""
Tests for DisclosureState
"""

import pytest
import numpy as np

from stgiii_core.disclosure import DisclosureState


class TestDisclosureState:
    """DisclosureStateの単体テスト"""

    def test_initial_state(self) -> None:
        """初期状態のテスト"""
        state = DisclosureState(n_total=100)
        assert state.n_disclosed == 0
        assert len(state.disclosed_indices) == 0
        assert len(state.disclosed_values) == 0
        assert state.get_undisclosed_count() == 100

    def test_disclose_single(self) -> None:
        """単一セル開示のテスト"""
        state = DisclosureState(n_total=100)
        new_count = state.disclose([5], [1.5])
        assert new_count == 1
        assert state.n_disclosed == 1
        assert state.is_disclosed(5)
        assert not state.is_disclosed(4)
        assert state.disclosed_values[0] == 1.5

    def test_disclose_multiple(self) -> None:
        """複数セル開示のテスト"""
        state = DisclosureState(n_total=100)
        new_count = state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert new_count == 3
        assert state.n_disclosed == 3
        assert state.is_disclosed(0)
        assert state.is_disclosed(5)
        assert state.is_disclosed(10)
        assert not state.is_disclosed(1)

    def test_disclose_duplicates(self) -> None:
        """重複開示のテスト（重複は無視される）"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5], [1.0, 2.0])
        new_count = state.disclose([5, 10], [2.0, 3.0])
        assert new_count == 1  # 5は既開示なのでカウントされない
        assert state.n_disclosed == 3

    def test_disclose_all_duplicates(self) -> None:
        """全て重複の開示"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5], [1.0, 2.0])
        new_count = state.disclose([0, 5], [1.0, 2.0])
        assert new_count == 0
        assert state.n_disclosed == 2

    def test_get_undisclosed_indices(self) -> None:
        """未開示セル取得のテスト"""
        state = DisclosureState(n_total=10)
        state.disclose([0, 5, 9], [1.0, 2.0, 3.0])
        undisclosed = state.get_undisclosed_indices()
        assert len(undisclosed) == 7
        assert 0 not in undisclosed
        assert 5 not in undisclosed
        assert 9 not in undisclosed
        assert 1 in undisclosed
        assert 8 in undisclosed

    def test_contains_any_true(self) -> None:
        """contains_any: いずれかが開示済み"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert state.contains_any([5, 20, 30])  # 5は開示済み
        assert state.contains_any([0])  # 0は開示済み

    def test_contains_any_false(self) -> None:
        """contains_any: 全て未開示"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert not state.contains_any([1, 2, 3])  # 全て未開示

    def test_contains_all_true(self) -> None:
        """contains_all: 全て開示済み"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert state.contains_all([0, 5, 10])
        assert state.contains_all([5])

    def test_contains_all_false(self) -> None:
        """contains_all: 一部が未開示"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        assert not state.contains_all([0, 5, 20])  # 20は未開示

    def test_get_data_for_training(self) -> None:
        """学習用データ取得のテスト"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        indices, values = state.get_data_for_training()
        assert len(indices) == 3
        assert len(values) == 3
        np.testing.assert_array_equal(indices, [0, 5, 10])
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_reset(self) -> None:
        """リセットのテスト"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        state.reset()
        assert state.n_disclosed == 0
        assert len(state.disclosed_indices) == 0
        assert not state.is_disclosed(0)

    def test_copy(self) -> None:
        """コピーのテスト"""
        state = DisclosureState(n_total=100)
        state.disclose([0, 5, 10], [1.0, 2.0, 3.0])
        copied = state.copy()

        # コピーは同じ状態
        assert copied.n_disclosed == 3
        assert copied.is_disclosed(5)

        # 元を変更してもコピーに影響しない
        state.disclose([20], [4.0])
        assert state.n_disclosed == 4
        assert copied.n_disclosed == 3
        assert not copied.is_disclosed(20)

    def test_length_mismatch_error(self) -> None:
        """indices/valuesの長さ不一致でエラー"""
        state = DisclosureState(n_total=100)
        with pytest.raises(ValueError, match="長さが一致しません"):
            state.disclose([0, 5], [1.0])  # 2つと1つ

    def test_disclosed_mask(self) -> None:
        """disclosed_maskプロパティのテスト"""
        state = DisclosureState(n_total=10)
        state.disclose([0, 5, 9], [1.0, 2.0, 3.0])
        mask = state.disclosed_mask
        assert mask[0] == True
        assert mask[1] == False
        assert mask[5] == True
        assert mask[9] == True
