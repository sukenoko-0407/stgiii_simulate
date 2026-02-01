"""
Integration tests for StageIII Simulator
"""

import pytest

from stgiii_core.config import SimulationConfig, SlotConfig, OperatorType
from stgiii_core.simulation import SimulationEngine


class TestIntegration:
    """統合テスト"""

    @pytest.mark.parametrize("n_slots", [2, 3, 4])
    def test_simulation_completes_all_slot_counts(
        self, n_slots: int
    ) -> None:
        """各スロット数でシミュレーションが完了すること"""
        slots = tuple(
            SlotConfig(name, 10)
            for name in ["A", "B", "C", "D"][:n_slots]
        )

        config = SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=3,
            slots=slots,
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=5,
            random_seed=42,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        for trial in results.trials:
            assert trial.p_top1 > 0
            assert trial.p_topk > 0
            assert trial.p_topk <= trial.p_top1
            assert trial.n_initial_disclosed > 0

    @pytest.mark.parametrize("operator_type", list(OperatorType))
    def test_all_operators_complete(
        self, operator_type: OperatorType
    ) -> None:
        """全Operatorが動作すること"""
        config = SimulationConfig(
            operator_type=operator_type,
            n_trials=3,
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

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        df = results.to_dataframe()
        assert len(df) == 3
        assert all(df["method"] == operator_type.value)

    @pytest.mark.parametrize("k_per_step", [1, 2, 3, 4, 5])
    def test_various_k_values(self, k_per_step: int) -> None:
        """各K値で動作すること"""
        config = SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=3,
            slots=(
                SlotConfig("A", 10),
                SlotConfig("B", 10),
            ),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=k_per_step,
            topk_k=5,
            random_seed=42,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        for trial in results.trials:
            assert trial.k_value == k_per_step

    @pytest.mark.parametrize("topk_k", [5, 10, 20])
    def test_various_topk_values(self, topk_k: int) -> None:
        """各Top-k値で動作すること"""
        config = SimulationConfig(
            operator_type=OperatorType.RANDOM,
            n_trials=3,
            slots=(
                SlotConfig("A", 15),
                SlotConfig("B", 15),
            ),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=1,
            topk_k=topk_k,
            random_seed=42,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        for trial in results.trials:
            assert trial.topk_k == topk_k

    def test_results_statistics(self) -> None:
        """結果の統計量計算"""
        config = SimulationConfig(
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

        engine = SimulationEngine(config)
        results = engine.run()

        stats = results.compute_statistics()

        assert "P_top1" in stats
        assert "P_topk" in stats
        assert "n_steps" in stats

        for key in ["P_top1", "P_topk"]:
            assert "median" in stats[key]
            assert "mean" in stats[key]
            assert "std" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]

    def test_results_to_csv(self, tmp_path) -> None:
        """CSVエクスポート"""
        config = SimulationConfig(
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

        engine = SimulationEngine(config)
        results = engine.run()

        csv_path = tmp_path / "results.csv"
        results.to_csv(str(csv_path))

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "P_top1" in content
        assert "P_topk" in content

    def test_uneven_slot_sizes(self) -> None:
        """不均一なスロットサイズ"""
        config = SimulationConfig(
            operator_type=OperatorType.FW_RIDGE,
            n_trials=3,
            slots=(
                SlotConfig("A", 10),
                SlotConfig("B", 20),
                SlotConfig("C", 15),
            ),
            main_effect_range=(-1.0, 1.0),
            error_clip_range=(-0.5, 0.5),
            k_per_step=2,
            topk_k=10,
            random_seed=42,
        )

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        assert results.config_summary["n_total_cells"] == 10 * 20 * 15

    def test_bayesian_operator_with_small_data(self) -> None:
        """ベイジアンOperatorが少量データで動作すること"""
        config = SimulationConfig(
            operator_type=OperatorType.BAYESIAN_FW_UCB,
            n_trials=3,
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

        engine = SimulationEngine(config)
        results = engine.run()

        assert len(results.trials) == 3
        # ベイジアンは一般にランダムより効率的
        # （ただし試行回数が少ないので統計的には有意でないかもしれない）


class TestConfigValidation:
    """設定バリデーションのテスト"""

    def test_slot_count_validation(self) -> None:
        """スロット数のバリデーション"""
        # 1スロットはエラー
        with pytest.raises(ValueError, match="スロット数は2〜4"):
            SimulationConfig(
                operator_type=OperatorType.RANDOM,
                n_trials=10,
                slots=(SlotConfig("A", 10),),
                main_effect_range=(-1.0, 1.0),
                error_clip_range=(-0.5, 0.5),
                k_per_step=1,
                topk_k=5,
            )

        # 5スロットはエラー
        with pytest.raises(ValueError, match="スロット数は2〜4"):
            SimulationConfig(
                operator_type=OperatorType.RANDOM,
                n_trials=10,
                slots=(
                    SlotConfig("A", 10),
                    SlotConfig("B", 10),
                    SlotConfig("C", 10),
                    SlotConfig("D", 10),
                    SlotConfig("E", 10),
                ),
                main_effect_range=(-1.0, 1.0),
                error_clip_range=(-0.5, 0.5),
                k_per_step=1,
                topk_k=5,
            )

    def test_total_cells_limit(self) -> None:
        """総セル数上限のバリデーション"""
        with pytest.raises(ValueError, match="総セル数が上限を超えています"):
            SimulationConfig(
                operator_type=OperatorType.RANDOM,
                n_trials=10,
                slots=(
                    SlotConfig("A", 50),
                    SlotConfig("B", 50),
                    SlotConfig("C", 50),
                ),
                main_effect_range=(-1.0, 1.0),
                error_clip_range=(-0.5, 0.5),
                k_per_step=1,
                topk_k=5,
            )  # 50*50*50 = 125,000 > 100,000

    def test_main_effect_range_validation(self) -> None:
        """主作用範囲のバリデーション"""
        with pytest.raises(ValueError, match="主作用範囲の下限は上限より小さい"):
            SimulationConfig(
                operator_type=OperatorType.RANDOM,
                n_trials=10,
                slots=(SlotConfig("A", 10), SlotConfig("B", 10)),
                main_effect_range=(1.0, -1.0),  # 逆順
                error_clip_range=(-0.5, 0.5),
                k_per_step=1,
                topk_k=5,
            )

    def test_error_range_validation(self) -> None:
        """誤差範囲のバリデーション"""
        with pytest.raises(ValueError, match="誤差範囲の下限は上限より小さい"):
            SimulationConfig(
                operator_type=OperatorType.RANDOM,
                n_trials=10,
                slots=(SlotConfig("A", 10), SlotConfig("B", 10)),
                main_effect_range=(-1.0, 1.0),
                error_clip_range=(0.5, -0.5),  # 逆順
                k_per_step=1,
                topk_k=5,
            )
