"""
Main entry point for CLI Simulator.

Usage:
    python -m cli_simulator [options]

Example:
    python -m cli_simulator --slots 20,20,20 --trials 100 --preset balanced
"""

import argparse
import sys

from .config_builder import build_base_config, format_config_summary, PRESETS
from .runner import run_all_strategies
from .reporter import print_full_report, export_to_csv, export_detailed_csv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="cli_simulator",
        description="StageIII CLI Simulator - Compare all strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli_simulator
  python -m cli_simulator --slots 25,30,20 --preset hard
  python -m cli_simulator --trials 200 --output results.csv
  python -m cli_simulator --slots 20,20,20,20 --f-main 0.4 --f-int 0.3 --f-res 0.3
        """
    )

    # Basic arguments
    parser.add_argument(
        "-s", "--slots",
        type=str,
        default="20,20,20",
        help="BB counts per slot, comma-separated (default: 20,20,20)"
    )
    parser.add_argument(
        "-t", "--trials",
        type=int,
        default=100,
        help="Number of trials per strategy (default: 100)"
    )
    parser.add_argument(
        "-k", "--k-per-step",
        type=int,
        default=1,
        help="Cells to disclose per step (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV file path"
    )

    # Difficulty preset
    parser.add_argument(
        "-p", "--preset",
        type=str,
        choices=["easy", "balanced", "hard"],
        default="balanced",
        help="Difficulty preset (default: balanced)"
    )

    # Generation model overrides
    parser.add_argument(
        "--f-main",
        type=float,
        default=None,
        help="Main effect variance ratio (overrides preset)"
    )
    parser.add_argument(
        "--f-int",
        type=float,
        default=None,
        help="Interaction variance ratio (overrides preset)"
    )
    parser.add_argument(
        "--f-res",
        type=float,
        default=None,
        help="Residual variance ratio (overrides preset)"
    )
    parser.add_argument(
        "--eta-spike",
        type=float,
        default=None,
        help="Activity cliff contribution (0-1)"
    )
    parser.add_argument(
        "--spike-hotspots",
        type=int,
        default=None,
        help="Number of hotspots per slot pair"
    )
    parser.add_argument(
        "--residual-nu",
        type=float,
        default=None,
        help="Residual t-distribution degrees of freedom"
    )
    parser.add_argument(
        "--distance-lambda",
        type=float,
        default=None,
        help="Slot distance scale parameter"
    )

    # Advanced: Continuous interaction operator settings
    parser.add_argument(
        "--operator-high-dim",
        type=int,
        default=256,
        choices=[256, 512],
        help="High-dim transformation output dimension (default: 256)"
    )
    parser.add_argument(
        "--operator-pca-dim",
        type=int,
        default=16,
        choices=[4, 8, 16, 32, 64],
        help="PCA dimension (default: 16)"
    )
    parser.add_argument(
        "--operator-mlp-hidden-dim",
        type=int,
        default=64,
        choices=[4, 8, 16, 32, 64],
        help="MLP hidden dimension (default: 64)"
    )
    parser.add_argument(
        "--operator-nonlinearity",
        type=str,
        default="tanh",
        choices=["tanh", "gelu"],
        help="Nonlinearity type (default: tanh)"
    )
    parser.add_argument(
        "--continuous-model",
        type=str,
        default="kron",
        choices=["kron", "low_rank"],
        help="Continuous interaction model (default: kron)"
    )
    parser.add_argument(
        "--continuous-rank",
        type=int,
        default=4,
        help="Low-rank model rank (default: 4)"
    )

    # Output control
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bars"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        # Build base configuration
        config = build_base_config(args)

        # Validate cell count
        if config.n_total_cells > 20_000:
            print(f"Error: Total cells ({config.n_total_cells:,}) exceeds limit (20,000)")
            print("Please reduce the number of building blocks per slot.")
            return 1

        # Print configuration
        print("\n" + "=" * 80)
        print("StageIII CLI Simulator")
        print("=" * 80)
        print("\nConfiguration:")
        print(format_config_summary(config, args.preset))
        print("\nRunning simulations for all 11 strategies...")
        print("=" * 80)

        # Run all strategies
        results = run_all_strategies(
            base_config=config,
            show_progress=not args.no_progress,
            verbose=args.verbose,
        )

        # Print results
        print("\n")
        print_full_report(results, config, args.preset)

        # Export to CSV if requested
        if args.output:
            export_to_csv(results, args.output)
            export_detailed_csv(results, args.output)

        return 0

    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
