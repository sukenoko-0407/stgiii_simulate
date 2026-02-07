"""
Runner for executing simulations across all strategies.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

from .stgiii_core.config import SimulationConfig, OperatorType
from .stgiii_core.simulation import SimulationEngine
from .stgiii_core.results import SimulationResults
from .config_builder import get_config_for_strategy, get_all_strategies


@dataclass
class StrategyResult:
    """Result for a single strategy."""
    strategy: OperatorType
    results: SimulationResults
    error: Optional[str] = None


def run_all_strategies(
    base_config: SimulationConfig,
    show_progress: bool = True,
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[OperatorType, StrategyResult]:
    """
    Run simulations for all strategies.

    Args:
        base_config: Base configuration (operator_type will be replaced)
        show_progress: Whether to show progress output
        verbose: Whether to show detailed output
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping OperatorType to StrategyResult
    """
    strategies = get_all_strategies()
    n_strategies = len(strategies)
    results: Dict[OperatorType, StrategyResult] = {}

    for i, strategy in enumerate(strategies, 1):
        strategy_name = strategy.name

        if show_progress:
            print(f"\n[{i}/{n_strategies}] Running {strategy.value}...", flush=True)

        if progress_callback:
            progress_callback(strategy_name, i, n_strategies)

        try:
            # Create config for this strategy
            config = get_config_for_strategy(base_config, strategy)

            # Create and run simulation engine
            def trial_progress(trial_id: int, n_trials: int) -> None:
                if show_progress and not verbose:
                    # Simple progress indicator
                    pct = (trial_id + 1) / n_trials * 100
                    bar_len = 30
                    filled = int(bar_len * (trial_id + 1) / n_trials)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    print(f"\r  Progress: [{bar}] {pct:5.1f}% ({trial_id + 1}/{n_trials})", end="", flush=True)

            engine = SimulationEngine(config, progress_callback=trial_progress if show_progress else None)
            sim_results = engine.run()

            if show_progress:
                print()  # New line after progress bar

            results[strategy] = StrategyResult(
                strategy=strategy,
                results=sim_results,
            )

            if verbose:
                stats = sim_results.compute_statistics()
                print(f"  -> P_top1 median: {stats['P_top1']['median']:.0f}, "
                      f"P_top100_50 median: {stats['P_top100_50']['median']:.0f}")

        except Exception as e:
            error_msg = str(e)
            if show_progress:
                print(f"  ERROR: {error_msg}")
            results[strategy] = StrategyResult(
                strategy=strategy,
                results=None,
                error=error_msg,
            )

    return results


def get_sorted_results(
    results: Dict[OperatorType, StrategyResult],
    sort_by: str = "P_top1_median"
) -> List[StrategyResult]:
    """
    Sort results by the specified metric.

    Args:
        results: Dictionary of strategy results
        sort_by: Metric to sort by (P_top1_median, P_top100_50_median)

    Returns:
        Sorted list of StrategyResult objects
    """
    def get_sort_key(sr: StrategyResult) -> float:
        if sr.error or sr.results is None:
            return float('inf')
        stats = sr.results.compute_statistics()
        if sort_by == "P_top1_median":
            return stats["P_top1"]["median"]
        elif sort_by == "P_top100_50_median":
            return stats["P_top100_50"]["median"]
        else:
            return stats["P_top1"]["median"]

    sorted_results = sorted(results.values(), key=get_sort_key)
    return sorted_results
