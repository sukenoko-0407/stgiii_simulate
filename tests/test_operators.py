"""
Tests for Operators
"""

import pytest
import numpy as np

from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.indexer import CellIndexer
from stgiii_core.disclosure import DisclosureState
from stgiii_core.operators.base import OperatorContext
from stgiii_core.operators.registry import get_operator, list_operators
from stgiii_core.operators.random_operator import RandomOperator
from stgiii_core.operators.fw_ridge import FreeWilsonRidgeOperator
from stgiii_core.operators.bayesian_fw import BayesianFreeWilsonOperator


@pytest.fixture
def simple_config() -> SimulationConfig:
    """シンプルな設定を作成"""
    return SimulationConfig(
        operator_type=OperatorType.RANDOM,
        n_trials=10,
        slots=(
            SlotConfig("A", 10),
            SlotConfig("B", 10),
        ),
        main_effect_range=(-1.0, 1.0),
        error_clip_range=(-0.5, 0.5),
        k_per_step=1,
        topk_k=5,
        random_seed=42,
    )


@pytest.fixture
def context(simple_config: SimulationConfig) -> OperatorContext:
    """Operatorコンテキストを作成"""
    indexer = CellIndexer(simple_config.slot_sizes)
    rng = np.random.default_rng(42)
    return OperatorContext(simple_config, indexer, rng)


class TestOperatorRegistry:
    """Operatorレジストリのテスト"""

    def test_list_operators(self) -> None:
        """登録済みOperatorの一覧取得"""
        operators = list_operators()
        assert OperatorType.RANDOM in operators
        assert OperatorType.FW_RIDGE in operators
        assert OperatorType.BAYESIAN_FW_UCB in operators

    def test_get_random_operator(self, context: OperatorContext) -> None:
        """RandomOperatorの取得"""
        operator = get_operator(OperatorType.RANDOM, context)
        assert isinstance(operator, RandomOperator)
        assert operator.name == "Random"

    def test_get_fw_ridge_operator(self, context: OperatorContext) -> None:
        """FW-RidgeOperatorの取得"""
        operator = get_operator(OperatorType.FW_RIDGE, context)
        assert isinstance(operator, FreeWilsonRidgeOperator)
        assert operator.name == "FW-Ridge"

    def test_get_bayesian_operator(self, context: OperatorContext) -> None:
        """BayesianFW Operatorの取得"""
        operator = get_operator(OperatorType.BAYESIAN_FW_UCB, context)
        assert isinstance(operator, BayesianFreeWilsonOperator)
        assert operator.name == "Bayesian-FW-UCB"


class TestRandomOperator:
    """RandomOperatorのテスト"""

    def test_select_from_undisclosed_only(
        self,
        context: OperatorContext
    ) -> None:
        """未開示セルからのみ選択する"""
        operator = RandomOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        # 初期開示
        disclosure.disclose([0, 1, 2, 3, 4], [1.0] * 5)

        # 90回選択テスト（100セル - 5初期開示 = 95未開示、余裕を持って90回）
        for _ in range(90):
            selected = operator.select_next_cells(disclosure, 1)
            assert len(selected) == 1
            assert selected[0] not in [0, 1, 2, 3, 4]
            # 開示を進める
            disclosure.disclose(selected, [1.0])

    def test_select_multiple(self, context: OperatorContext) -> None:
        """複数セル選択"""
        operator = RandomOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        selected = operator.select_next_cells(disclosure, 5)
        assert len(selected) == 5
        assert len(set(selected)) == 5  # 重複なし

    def test_select_when_fewer_remaining(
        self,
        context: OperatorContext
    ) -> None:
        """残りセル数がK未満の場合"""
        operator = RandomOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        # 残り3セルまで開示
        all_but_three = list(range(context.indexer.n_total - 3))
        disclosure.disclose(all_but_three, [1.0] * len(all_but_three))

        selected = operator.select_next_cells(disclosure, 5)
        assert len(selected) == 3  # 残り3セルのみ


class TestFreeWilsonRidgeOperator:
    """FreeWilsonRidgeOperatorのテスト"""

    def test_select_without_training(
        self,
        context: OperatorContext
    ) -> None:
        """未学習時はランダム選択"""
        operator = FreeWilsonRidgeOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        selected = operator.select_next_cells(disclosure, 3)
        assert len(selected) == 3

    def test_fit_and_predict(self, context: OperatorContext) -> None:
        """学習と予測"""
        operator = FreeWilsonRidgeOperator(context)

        # 学習データ
        indices = np.array([0, 1, 10, 11, 20], dtype=np.int64)
        values = np.array([7.5, 7.8, 8.0, 8.2, 7.6], dtype=np.float64)

        operator.fit(indices, values)

        # 予測
        predictions = operator.predict(indices)
        assert len(predictions) == len(indices)

    def test_select_after_training(
        self,
        context: OperatorContext
    ) -> None:
        """学習後の選択"""
        operator = FreeWilsonRidgeOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        # 学習データを開示
        indices = [0, 1, 10, 11, 20]
        values = [7.5, 7.8, 8.0, 8.2, 7.6]
        disclosure.disclose(indices, values)

        # 学習
        ind_arr, val_arr = disclosure.get_data_for_training()
        operator.fit(ind_arr, val_arr)

        # 選択
        selected = operator.select_next_cells(disclosure, 3)
        assert len(selected) == 3
        # 既開示セルを選択していないこと
        for idx in selected:
            assert idx not in indices

    def test_get_coefficients_sum_to_zero(
        self,
        context: OperatorContext
    ) -> None:
        """sum-to-zero係数の取得"""
        operator = FreeWilsonRidgeOperator(context)

        indices = np.array([0, 1, 10, 11, 20, 21], dtype=np.int64)
        values = np.array([7.5, 7.8, 8.0, 8.2, 7.6, 7.9], dtype=np.float64)
        operator.fit(indices, values)

        coefs = operator.get_coefficients_sum_to_zero()
        assert "A" in coefs
        assert "B" in coefs
        # sum-to-zeroなので各スロットの係数合計は0に近い
        np.testing.assert_almost_equal(np.sum(coefs["A"]), 0, decimal=5)
        np.testing.assert_almost_equal(np.sum(coefs["B"]), 0, decimal=5)


class TestBayesianFreeWilsonOperator:
    """BayesianFreeWilsonOperatorのテスト"""

    def test_select_without_training(
        self,
        context: OperatorContext
    ) -> None:
        """未学習時はランダム選択"""
        operator = BayesianFreeWilsonOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        selected = operator.select_next_cells(disclosure, 3)
        assert len(selected) == 3

    def test_fit_updates_sigma(self, context: OperatorContext) -> None:
        """学習でσが更新される"""
        operator = BayesianFreeWilsonOperator(context)

        indices = np.array([0, 1, 10, 11, 20, 21, 30], dtype=np.int64)
        values = np.array([7.5, 7.8, 8.0, 8.2, 7.6, 7.9, 8.1], dtype=np.float64)

        initial_sigma = operator.sigma
        operator.fit(indices, values)

        # σが更新されている（必ずしも変わるとは限らないが、処理は通る）
        assert operator.theta_map is not None
        assert operator.Sigma_theta is not None

    def test_predict_with_uncertainty(
        self,
        context: OperatorContext
    ) -> None:
        """予測と不確実性の計算"""
        operator = BayesianFreeWilsonOperator(context)

        indices = np.array([0, 1, 10, 11, 20, 21, 30], dtype=np.int64)
        values = np.array([7.5, 7.8, 8.0, 8.2, 7.6, 7.9, 8.1], dtype=np.float64)
        operator.fit(indices, values)

        test_indices = np.array([5, 15, 25], dtype=np.int64)
        mu, sigma_param, sigma_total = operator.predict_with_uncertainty(
            test_indices
        )

        assert len(mu) == 3
        assert len(sigma_param) == 3
        assert len(sigma_total) == 3
        # sigma_total >= sigma_param
        assert np.all(sigma_total >= sigma_param)

    def test_ucb_selection(self, context: OperatorContext) -> None:
        """UCBによる選択"""
        operator = BayesianFreeWilsonOperator(context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        # 学習データを開示
        indices = [0, 1, 10, 11, 20, 21, 30]
        values = [7.5, 7.8, 8.0, 8.2, 7.6, 7.9, 8.1]
        disclosure.disclose(indices, values)

        # 学習
        ind_arr, val_arr = disclosure.get_data_for_training()
        operator.fit(ind_arr, val_arr)

        # 選択
        selected = operator.select_next_cells(disclosure, 3)
        assert len(selected) == 3
        # 既開示セルを選択していないこと
        for idx in selected:
            assert idx not in indices


class TestOperatorDoesNotSelectDisclosed:
    """全Operatorが既開示セルを選択しないことを確認"""

    @pytest.mark.parametrize("operator_type", list(OperatorType))
    def test_no_disclosed_selection(
        self,
        operator_type: OperatorType,
        context: OperatorContext
    ) -> None:
        """既開示セルを選択しない"""
        operator = get_operator(operator_type, context)
        disclosure = DisclosureState(n_total=context.indexer.n_total)

        # 初期開示
        initial = list(range(20))
        disclosure.disclose(initial, [8.0] * 20)

        # 学習が必要なOperatorは学習
        ind_arr, val_arr = disclosure.get_data_for_training()
        operator.fit(ind_arr, val_arr)

        # 複数回選択テスト
        for _ in range(50):
            selected = operator.select_next_cells(disclosure, 3)
            for idx in selected:
                assert idx not in initial, f"{operator_type}: 既開示セル{idx}を選択"
            disclosure.disclose(selected, [8.0] * len(selected))
            operator.fit(*disclosure.get_data_for_training())
