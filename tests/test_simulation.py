"""
Tests for SimulationEngine
"""

import pytest
import numpy as np

from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.simulation import SimulationEngine
from stgiii_core.matrix import MatrixGenerator
from stgiii_core.indexer import CellIndexer


@pytest.fixture
def simple_config() -> SimulationConfig:
    """シンプルな設定を作成"""
    return SimulationConfig(
        operator_type=OperatorType.RANDOM,
        n_trials=5,
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


class TestSimulationEngine:
    """SimulationEngineのテスト"""

    def test_calculate_initial_disclosure_count_2slots(self) -> None:
        """初期開示セル数計算（2スロット）"""
        count = SimulationEngine.calculate_initial_disclosure_count((10, 20))
        # 10 + 20 - (2-1) = 29
        assert count == 29

    def test_calculate_initial_disclosure_count_3slots(self) -> None:
        """初期開示セル数計算（3スロット）"""
        count = SimulationEngine.calculate_initial_disclosure_count((10, 20, 15))
        # 10 + 20 + 15 - (3-1) = 43
        assert count == 43

    def test_calculate_initial_disclosure_count_4slots(self) -> None:
        """初期開示セル数計算（4スロット）"""
        count = SimulationEngine.calculate_initial_disclosure_count((10, 10, 10, 10))
        # 10*4 - 3 = 37
        assert count == 37

    def test_run_completes(self, simple_config: SimulationConfig) -> None:
        """シミュレーションが完了すること"""
        engine = SimulationEngine(simple_config)
        results = engine.run()

        assert len(results.trials) == simple_config.n_trials
        for trial in results.trials:
            assert trial.p_top1 > 0
            assert trial.p_topk > 0
            assert trial.p_topk <= trial.p_top1

    def test_progress_callback(self, simple_config: SimulationConfig) -> None:
        """進捗コールバックが呼ばれること"""
        callback_calls = []

        def callback(current: int, total: int) -> None:
            callback_calls.append((current, total))

        engine = SimulationEngine(simple_config, progress_callback=callback)
        engine.run()

        # n_trials + 1 回呼ばれる（0からn_trialsまで）
        assert len(callback_calls) == simple_config.n_trials + 1
        assert callback_calls[0] == (0, simple_config.n_trials)
        assert callback_calls[-1] == (
            simple_config.n_trials, simple_config.n_trials
        )

    def test_reproducibility_with_seed(self) -> None:
        """シードによる再現性"""
        config = SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=5,
            slots=(SlotConfig("A", 10), SlotConfig("B", 10)),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=5,
            random_seed=12345,
        )

        engine1 = SimulationEngine(config)
        results1 = engine1.run()

        engine2 = SimulationEngine(config)
        results2 = engine2.run()

        # 同じシードなら同じ結果
        for t1, t2 in zip(results1.trials, results2.trials):
            assert t1.p_top1 == t2.p_top1
            assert t1.p_topk == t2.p_topk

    def test_validate_config_no_warnings(
        self, simple_config: SimulationConfig
    ) -> None:
        """正常な設定で警告なし"""
        warnings = SimulationEngine.validate_config(simple_config)
        assert len(warnings) == 0


class TestMatrixGenerator:
    """MatrixGeneratorのテスト"""

    def test_generate_produces_valid_matrix(
        self, simple_config: SimulationConfig
    ) -> None:
        """有効なMatrixを生成"""
        indexer = CellIndexer(simple_config.slot_sizes)
        rng = np.random.default_rng(42)
        generator = MatrixGenerator(simple_config, indexer, rng)

        matrix = generator.generate()

        assert len(matrix.y_true) == indexer.n_total
        assert len(matrix.y_obs) == indexer.n_total
        assert matrix.top1_index >= 0
        assert matrix.top1_index < indexer.n_total
        assert len(matrix.topk_indices) == simple_config.topk_k

    def test_y_obs_is_clipped(self, simple_config: SimulationConfig) -> None:
        """y_obsがclipされていること"""
        indexer = CellIndexer(simple_config.slot_sizes)
        rng = np.random.default_rng(42)
        generator = MatrixGenerator(simple_config, indexer, rng)

        matrix = generator.generate()

        obs_min, obs_max = simple_config.obs_clip_range
        assert np.all(matrix.y_obs >= obs_min)
        assert np.all(matrix.y_obs <= obs_max)

    def test_top1_is_unique_argmax(
        self, simple_config: SimulationConfig
    ) -> None:
        """top1_indexがy_trueの一意なargmax"""
        indexer = CellIndexer(simple_config.slot_sizes)
        rng = np.random.default_rng(42)
        generator = MatrixGenerator(simple_config, indexer, rng)

        matrix = generator.generate()

        assert matrix.top1_index == np.argmax(matrix.y_true)

    def test_topk_are_top_values(
        self, simple_config: SimulationConfig
    ) -> None:
        """topk_indicesがy_trueの上位k個"""
        indexer = CellIndexer(simple_config.slot_sizes)
        rng = np.random.default_rng(42)
        generator = MatrixGenerator(simple_config, indexer, rng)

        matrix = generator.generate()

        # topk_indicesの値がy_trueの上位k個に含まれる
        sorted_indices = np.argsort(matrix.y_true)[::-1]
        expected_topk = set(sorted_indices[:simple_config.topk_k])
        actual_topk = set(matrix.topk_indices)
        assert actual_topk == expected_topk
