"""
Configuration builder for CLI Simulator.

Converts command-line arguments to SimulationConfig.
"""

from argparse import Namespace
from dataclasses import replace
from typing import Dict, Any

from .stgiii_core.config import (
    SimulationConfig,
    SlotConfig,
    OperatorType,
    NonlinearityType,
    ContinuousInteractionModel,
    InitialDisclosureType,
)


# Difficulty presets (matching WebApp sidebar.py)
PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "f_main": 0.6,
        "f_int": 0.2,
        "f_res": 0.2,
        "eta_spike": 0.15,
        "spike_hotspots": 1,
        "residual_nu": 10.0,
        "distance_lambda": 1.5,
    },
    "balanced": {
        "f_main": 0.35,
        "f_int": 0.35,
        "f_res": 0.30,
        "eta_spike": 0.30,
        "spike_hotspots": 2,
        "residual_nu": 6.0,
        "distance_lambda": 1.0,
    },
    "hard": {
        "f_main": 0.2,
        "f_int": 0.3,
        "f_res": 0.5,
        "eta_spike": 0.5,
        "spike_hotspots": 3,
        "residual_nu": 4.0,
        "distance_lambda": 0.5,
    },
}


def parse_slots(slots_str: str) -> tuple:
    """
    Parse slots string like "20,20,20" into SlotConfig tuple.

    Args:
        slots_str: Comma-separated BB counts (e.g., "20,25,30")

    Returns:
        Tuple of SlotConfig objects
    """
    slot_names = ["A", "B", "C", "D"]
    bb_counts = [int(x.strip()) for x in slots_str.split(",")]

    if len(bb_counts) < 2 or len(bb_counts) > 4:
        raise ValueError(f"Slot count must be 2-4, got {len(bb_counts)}")

    return tuple(
        SlotConfig(name=slot_names[i], n_building_blocks=count)
        for i, count in enumerate(bb_counts)
    )


def build_base_config(args: Namespace) -> SimulationConfig:
    """
    Build a base SimulationConfig from command-line arguments.

    The operator_type will be set to RANDOM as a placeholder;
    runner.py will replace it for each strategy.

    Args:
        args: Parsed command-line arguments

    Returns:
        SimulationConfig with all parameters set
    """
    # Parse slots
    slots = parse_slots(args.slots)

    # Start with preset values
    preset = PRESETS.get(args.preset, PRESETS["balanced"])

    # Override with explicit arguments if provided
    f_main = args.f_main if args.f_main is not None else preset["f_main"]
    f_int = args.f_int if args.f_int is not None else preset["f_int"]
    f_res = args.f_res if args.f_res is not None else preset["f_res"]
    eta_spike = args.eta_spike if args.eta_spike is not None else preset["eta_spike"]
    spike_hotspots = args.spike_hotspots if args.spike_hotspots is not None else preset["spike_hotspots"]
    residual_nu = args.residual_nu if args.residual_nu is not None else preset["residual_nu"]
    distance_lambda = args.distance_lambda if args.distance_lambda is not None else preset["distance_lambda"]

    # Normalize f values to sum to 1.0
    f_sum = f_main + f_int + f_res
    if abs(f_sum - 1.0) > 1e-6:
        f_main = f_main / f_sum
        f_int = f_int / f_sum
        f_res = f_res / f_sum

    # Parse nonlinearity
    nonlinearity = NonlinearityType.TANH
    if hasattr(args, 'operator_nonlinearity') and args.operator_nonlinearity:
        if args.operator_nonlinearity.lower() == "gelu":
            nonlinearity = NonlinearityType.GELU

    # Parse continuous model
    continuous_model = ContinuousInteractionModel.KRON
    if hasattr(args, 'continuous_model') and args.continuous_model:
        if args.continuous_model.lower() == "low_rank":
            continuous_model = ContinuousInteractionModel.LOW_RANK

    # Build config (operator_type is placeholder, will be replaced per strategy)
    config = SimulationConfig(
        operator_type=OperatorType.RANDOM,  # Placeholder
        n_trials=args.trials,
        slots=slots,
        k_per_step=args.k_per_step,
        topk_k=100,
        initial_disclosure_type=InitialDisclosureType.NONE,
        random_seed=args.seed,

        # Operator extension parameters
        operator_high_dim=getattr(args, 'operator_high_dim', 256),
        operator_pca_dim=getattr(args, 'operator_pca_dim', 16),
        operator_mlp_hidden_dim=getattr(args, 'operator_mlp_hidden_dim', 64),
        operator_nonlinearity=nonlinearity,
        continuous_interaction_model=continuous_model,
        continuous_interaction_rank=getattr(args, 'continuous_rank', 4),

        # Generation model parameters
        f_main=f_main,
        f_int=f_int,
        f_res=f_res,
        distance_lambda=distance_lambda,
        eta_spike=eta_spike,
        spike_hotspots=spike_hotspots,
        residual_nu=residual_nu,
    )

    return config


def get_config_for_strategy(
    base_config: SimulationConfig,
    strategy: OperatorType
) -> SimulationConfig:
    """
    Create a new config with the specified operator type.

    Args:
        base_config: Base configuration with all parameters
        strategy: The operator type to use

    Returns:
        New SimulationConfig with the specified operator type
    """
    return replace(base_config, operator_type=strategy)


def get_all_strategies() -> list:
    """
    Get list of all available strategies (OperatorType values).

    Returns:
        List of all OperatorType enum values
    """
    return list(OperatorType)


def format_config_summary(config: SimulationConfig, preset_name: str) -> str:
    """
    Format configuration summary for display.

    Args:
        config: SimulationConfig to summarize
        preset_name: Name of the preset used

    Returns:
        Formatted string summary
    """
    slot_desc = " × ".join(
        f"{s.name}({s.n_building_blocks})" for s in config.slots
    )

    lines = [
        f"  Slots: {slot_desc} = {config.n_total_cells:,} cells",
        f"  Trials: {config.n_trials}",
        f"  K per Step: {config.k_per_step}",
        f"  Preset: {preset_name} (f_main={config.f_main:.2f}, f_int={config.f_int:.2f}, f_res={config.f_res:.2f})",
        f"  eta_spike: {config.eta_spike:.2f}, spike_hotspots: {config.spike_hotspots}",
        f"  residual_nu: {config.residual_nu:.1f}, distance_lambda: {config.distance_lambda:.1f}",
    ]

    if config.random_seed is not None:
        lines.append(f"  Random Seed: {config.random_seed}")

    return "\n".join(lines)
