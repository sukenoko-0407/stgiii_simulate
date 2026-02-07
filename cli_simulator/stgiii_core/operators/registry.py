"""
Operator registry for StageIII Simulator

Operatorクラスの登録・取得機構（プラグインパターン）
"""

from typing import Type, Callable, Dict, List

from .base import BaseOperator, OperatorContext
from ..config import OperatorType


# Operatorクラスの登録用辞書
_OPERATOR_REGISTRY: Dict[OperatorType, Type[BaseOperator]] = {}


def register_operator(
    operator_type: OperatorType
) -> Callable[[Type[BaseOperator]], Type[BaseOperator]]:
    """
    Operatorクラスを登録するデコレータ

    Usage:
        @register_operator(OperatorType.RANDOM)
        class RandomOperator(BaseOperator):
            ...

    Args:
        operator_type: 登録するOperatorType

    Returns:
        デコレータ関数

    Raises:
        ValueError: 同じOperatorTypeが既に登録されている場合
    """
    def decorator(cls: Type[BaseOperator]) -> Type[BaseOperator]:
        if operator_type in _OPERATOR_REGISTRY:
            existing = _OPERATOR_REGISTRY[operator_type]
            raise ValueError(
                f"OperatorType {operator_type} is already registered by {existing.__name__}"
            )
        _OPERATOR_REGISTRY[operator_type] = cls
        return cls

    return decorator


def get_operator(
    operator_type: OperatorType,
    context: OperatorContext
) -> BaseOperator:
    """
    登録されたOperatorをインスタンス化して取得

    Args:
        operator_type: Operatorの種別
        context: Operatorコンテキスト

    Returns:
        初期化されたOperatorインスタンス

    Raises:
        ValueError: 未登録のOperatorType
    """
    if operator_type not in _OPERATOR_REGISTRY:
        available = list(_OPERATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown operator type: {operator_type}. "
            f"Available types: {available}"
        )

    cls = _OPERATOR_REGISTRY[operator_type]
    return cls(context)


def list_operators() -> List[OperatorType]:
    """
    登録済みのOperatorType一覧を取得

    Returns:
        登録済みOperatorTypeのリスト
    """
    return list(_OPERATOR_REGISTRY.keys())


def get_operator_class(operator_type: OperatorType) -> Type[BaseOperator]:
    """
    登録されたOperatorクラスを取得（インスタンス化なし）

    Args:
        operator_type: Operatorの種別

    Returns:
        Operatorクラス

    Raises:
        ValueError: 未登録のOperatorType
    """
    if operator_type not in _OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator type: {operator_type}")
    return _OPERATOR_REGISTRY[operator_type]


def is_registered(operator_type: OperatorType) -> bool:
    """
    指定したOperatorTypeが登録済みか確認

    Args:
        operator_type: 確認するOperatorType

    Returns:
        登録済みならTrue
    """
    return operator_type in _OPERATOR_REGISTRY
