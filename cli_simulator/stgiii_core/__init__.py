"""
StageIII Simulator Core Package

低分子創薬における組み合わせ合成ステージ（StageIII）シミュレーターのコアモジュール
"""

from .config import (
    SimulationConfig,
    SlotConfig,
    OperatorType,
    NonlinearityType,
    ContinuousInteractionModel,
)
from .indexer import CellIndexer
from .matrix import Matrix, MatrixGenerator, OperatorFeatureBundle
from .disclosure import DisclosureState
from .results import TrialResult, SimulationResults
from .simulation import SimulationEngine
from .exceptions import (
    StgIIISimulatorError,
    ConfigurationError,
    MatrixGenerationError,
    OperatorError,
    CellLimitExceededError,
    UniqueArgmaxError,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "SimulationConfig",
    "SlotConfig",
    "OperatorType",
    "NonlinearityType",
    "ContinuousInteractionModel",
    # Core
    "CellIndexer",
    "Matrix",
    "MatrixGenerator",
    "OperatorFeatureBundle",
    "DisclosureState",
    "TrialResult",
    "SimulationResults",
    "SimulationEngine",
    # Exceptions
    "StgIIISimulatorError",
    "ConfigurationError",
    "MatrixGenerationError",
    "OperatorError",
    "CellLimitExceededError",
    "UniqueArgmaxError",
]
