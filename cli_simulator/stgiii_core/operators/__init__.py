"""
Operator implementations for StageIII Simulator
"""

from .base import BaseOperator, OperatorContext, OperatorProtocol
from .registry import register_operator, get_operator, list_operators

# Import operators to trigger registration
from .random_operator import RandomOperator
from .fw_ols import FreeWilsonOLSOperator
from .fw_ridge import FreeWilsonRidgeOperator
from .bayesian_fw import BayesianFreeWilsonOperator, BayesianFreeWilsonTSOperator
from .fw_interactions import (
    FreeWilsonOLSDiscreteOperator,
    FreeWilsonRidgeDiscreteOperator,
    BayesianFreeWilsonDiscreteOperator,
    FreeWilsonOLSContinuousOperator,
    FreeWilsonRidgeContinuousOperator,
    BayesianFreeWilsonContinuousOperator,
)

__all__ = [
    "BaseOperator",
    "OperatorContext",
    "OperatorProtocol",
    "register_operator",
    "get_operator",
    "list_operators",
    "RandomOperator",
    "FreeWilsonOLSOperator",
    "FreeWilsonRidgeOperator",
    "BayesianFreeWilsonOperator",
    "BayesianFreeWilsonTSOperator",
    "FreeWilsonOLSDiscreteOperator",
    "FreeWilsonRidgeDiscreteOperator",
    "BayesianFreeWilsonDiscreteOperator",
    "FreeWilsonOLSContinuousOperator",
    "FreeWilsonRidgeContinuousOperator",
    "BayesianFreeWilsonContinuousOperator",
]
