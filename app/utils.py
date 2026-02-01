"""
Utility functions for Streamlit UI
"""

from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

# 日本語フォント設定（環境に応じて調整）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


def create_histogram(
    values: list[int] | NDArray[np.int64],
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int = 30,
    median_line: bool = True,
    figsize: Tuple[float, float] = (10, 5)
) -> Figure:
    """
    ヒストグラムを作成

    Args:
        values: データ値
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        bins: ビン数
        median_line: 中央値の縦線を表示するか
        figsize: 図のサイズ

    Returns:
        Matplotlibのfigureオブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    values_arr = np.array(values)
    ax.hist(values_arr, bins=bins, edgecolor="black", alpha=0.7, color="#4C78A8")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if median_line:
        median_val = np.median(values_arr)
        ax.axvline(
            median_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.1f}"
        )
        ax.legend()

    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    return fig


def format_number(value: float, decimals: int = 1) -> str:
    """
    数値をフォーマット

    Args:
        value: 数値
        decimals: 小数点以下桁数

    Returns:
        フォーマットされた文字列
    """
    if decimals == 0:
        return f"{value:,.0f}"
    else:
        return f"{value:,.{decimals}f}"


def create_comparison_histogram(
    values_list: list[list[int]],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int = 30,
    figsize: Tuple[float, float] = (12, 6)
) -> Figure:
    """
    複数のデータを比較するヒストグラムを作成

    Args:
        values_list: データ値のリスト（各手法ごと）
        labels: 各データのラベル
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        bins: ビン数
        figsize: 図のサイズ

    Returns:
        Matplotlibのfigureオブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']

    for i, (values, label) in enumerate(zip(values_list, labels)):
        values_arr = np.array(values)
        ax.hist(
            values_arr,
            bins=bins,
            edgecolor="black",
            alpha=0.5,
            color=colors[i % len(colors)],
            label=label
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    return fig
