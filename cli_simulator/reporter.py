"""
Reporter for displaying and exporting simulation results.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from .stgiii_core.config import SimulationConfig, OperatorType
from .runner import StrategyResult, get_sorted_results


SEPARATOR = "=" * 80
THIN_SEPARATOR = "-" * 80


def print_header() -> None:
    """Print report header."""
    print(SEPARATOR)
    print("                    StageIII CLI Simulator - Results Comparison")
    print(SEPARATOR)


def print_configuration(config: SimulationConfig, preset_name: str) -> None:
    """Print configuration summary."""
    slot_desc = " × ".join(
        f"{s.name}({s.n_building_blocks})" for s in config.slots
    )

    print("Configuration:")
    print(f"  Slots: {slot_desc} = {config.n_total_cells:,} cells")
    print(f"  Trials: {config.n_trials}")
    print(f"  K per Step: {config.k_per_step}")
    print(f"  Preset: {preset_name} (f_main={config.f_main:.2f}, f_int={config.f_int:.2f}, f_res={config.f_res:.2f})")

    if config.random_seed is not None:
        print(f"  Random Seed: {config.random_seed}")

    print(THIN_SEPARATOR)


def format_table_row(
    strategy_name: str,
    median: float,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
    error: Optional[str] = None
) -> str:
    """Format a single table row."""
    if error:
        return f"{strategy_name:<28} | {'ERROR':>6} | {error[:30]}"

    return (
        f"{strategy_name:<28} | {median:>6.0f} | {mean:>6.0f} | "
        f"{std:>6.0f} | {min_val:>6.0f} | {max_val:>6.0f} |"
    )


def print_comparison_table(
    results: Dict[OperatorType, StrategyResult],
    metric: str = "P_top1",
    title: str = "P_top1 - cells to find #1"
) -> None:
    """Print comparison table for a metric."""
    print(f"\nStrategy Comparison ({title}):")
    print(THIN_SEPARATOR)
    print(f"{'Strategy':<28} | {'Median':>6} | {'Mean':>6} | {'STD':>6} | {'Min':>6} | {'Max':>6} |")
    print(THIN_SEPARATOR)

    # Sort by median of the metric
    sorted_results = get_sorted_results(results, f"{metric}_median")

    for sr in sorted_results:
        if sr.error or sr.results is None:
            row = format_table_row(sr.strategy.name, 0, 0, 0, 0, 0, sr.error)
        else:
            stats = sr.results.compute_statistics()
            metric_stats = stats[metric]
            row = format_table_row(
                sr.strategy.name,
                metric_stats["median"],
                metric_stats["mean"],
                metric_stats["std"],
                metric_stats["min"],
                metric_stats["max"],
            )
        print(row)

    print(THIN_SEPARATOR)


def print_best_strategy(results: Dict[OperatorType, StrategyResult]) -> None:
    """Print the best strategy based on P_top1 median."""
    sorted_results = get_sorted_results(results, "P_top1_median")

    # Find first non-error result
    best = None
    for sr in sorted_results:
        if sr.error is None and sr.results is not None:
            best = sr
            break

    if best:
        stats = best.results.compute_statistics()
        median = stats["P_top1"]["median"]
        print(f"\nBest Strategy: {best.strategy.name} (Median P_top1: {median:.0f})")

    print(SEPARATOR)


def print_full_report(
    results: Dict[OperatorType, StrategyResult],
    config: SimulationConfig,
    preset_name: str
) -> None:
    """Print the full comparison report."""
    print_header()
    print_configuration(config, preset_name)

    # P_top1 table
    print_comparison_table(
        results,
        metric="P_top1",
        title="P_top1 - cells to find #1"
    )

    # P_top100_50 table
    print_comparison_table(
        results,
        metric="P_top100_50",
        title="P_top100_50 - cells to find 50 of top-100"
    )

    print_best_strategy(results)


def export_to_csv(
    results: Dict[OperatorType, StrategyResult],
    output_path: str
) -> None:
    """
    Export results to CSV file.

    Args:
        results: Dictionary of strategy results
        output_path: Path to output CSV file
    """
    path = Path(output_path)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'strategy', 'metric', 'median', 'mean', 'std', 'min', 'max'
        ])

        # Data rows
        for strategy, sr in results.items():
            if sr.error or sr.results is None:
                # Write error row
                writer.writerow([strategy.name, 'error', '', '', '', '', sr.error or 'Unknown error'])
                continue

            stats = sr.results.compute_statistics()

            for metric in ['P_top1', 'P_topk', 'P_top100_50']:
                if metric in stats:
                    metric_stats = stats[metric]
                    writer.writerow([
                        strategy.name,
                        metric,
                        f"{metric_stats['median']:.1f}",
                        f"{metric_stats['mean']:.1f}",
                        f"{metric_stats['std']:.1f}",
                        f"{metric_stats['min']:.1f}",
                        f"{metric_stats['max']:.1f}",
                    ])

    print(f"\nResults exported to: {path.absolute()}")


def export_detailed_csv(
    results: Dict[OperatorType, StrategyResult],
    output_path: str
) -> None:
    """
    Export detailed trial-level results to CSV.

    Args:
        results: Dictionary of strategy results
        output_path: Path to output CSV file
    """
    path = Path(output_path)
    base = path.stem
    ext = path.suffix
    detailed_path = path.parent / f"{base}_detailed{ext}"

    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'strategy', 'trial_id', 'p_top1', 'p_topk', 'p_top100_50',
            'n_steps', 'runtime_ms'
        ])

        # Data rows
        for strategy, sr in results.items():
            if sr.error or sr.results is None:
                continue

            for trial in sr.results.trials:
                writer.writerow([
                    strategy.name,
                    trial.trial_id,
                    trial.p_top1,
                    trial.p_topk,
                    trial.p_top100_50,
                    trial.n_steps,
                    f"{trial.runtime_ms:.1f}" if trial.runtime_ms else "",
                ])

    print(f"Detailed results exported to: {detailed_path.absolute()}")
