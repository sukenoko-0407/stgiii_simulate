"""
Result data structures for StageIII Simulator

試行結果とシミュレーション結果の集約
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass
class TrialResult:
    """単一試行の結果"""

    trial_id: int                  # 試行ID（0始まり）
    method: str                    # 使用した手法名
    n_total_cells: int             # 総セル数
    n_initial_disclosed: int       # 初期開示セル数
    k_value: int                   # 1ステップの開示数K
    topk_k: int                    # Top-kのk値
    p_top1: int                    # Top-1到達時の開示セル数
    p_topk: int                    # Top-k到達時の開示セル数
    p_top100_50: int               # Top-100のうち50個到達時の開示セル数
    n_steps: int                   # 反復ステップ数（初期開示除く）
    hit_in_initial_top1: bool      # 初期開示でTop-1到達したか
    hit_in_initial_topk: bool      # 初期開示でTop-k到達したか
    runtime_ms: float | None = None  # 処理時間（ミリ秒）

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "trial_id": self.trial_id,
            "method": self.method,
            "n_total_cells": self.n_total_cells,
            "n_initial_disclosed": self.n_initial_disclosed,
            "k_value": self.k_value,
            "topk_k": self.topk_k,
            "P_top1": self.p_top1,
            "P_topk": self.p_topk,
            "P_top100_50": self.p_top100_50,
            "n_steps": self.n_steps,
            "hit_in_initial_top1": self.hit_in_initial_top1,
            "hit_in_initial_topk": self.hit_in_initial_topk,
            "runtime_ms": self.runtime_ms,
        }


@dataclass
class SimulationResults:
    """全試行の結果を集約"""

    trials: List[TrialResult]
    config_summary: Dict[str, Any]

    @property
    def n_trials(self) -> int:
        """試行数"""
        return len(self.trials)

    def to_dataframe(self) -> pd.DataFrame:
        """
        結果をDataFrameに変換

        Returns:
            試行結果のDataFrame
        """
        records = [t.to_dict() for t in self.trials]
        return pd.DataFrame(records)

    def to_csv(self, path: str, index: bool = False) -> None:
        """
        結果をCSVに出力

        Args:
            path: 出力ファイルパス
            index: インデックスを出力するか
        """
        self.to_dataframe().to_csv(path, index=index)

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        統計量を計算

        Returns:
            {"P_top1": {...}, "P_topk": {...}, "P_top100_50": {...}} 形式の辞書
            各項目は median, mean, std, max, min を含む
        """
        df = self.to_dataframe()

        def calc_stats(series: pd.Series) -> Dict[str, float]:
            return {
                "median": float(series.median()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "max": float(series.max()),
                "min": float(series.min()),
            }

        return {
            "P_top1": calc_stats(df["P_top1"]),
            "P_topk": calc_stats(df["P_topk"]),
            "P_top100_50": calc_stats(df["P_top100_50"]),
            "n_steps": calc_stats(df["n_steps"]),
        }

    def get_summary_text(self) -> str:
        """
        結果のサマリテキストを生成

        Returns:
            人間可読なサマリ文字列
        """
        stats = self.compute_statistics()
        p1 = stats["P_top1"]
        pk = stats["P_topk"]
        p100_50 = stats["P_top100_50"]

        lines = [
            f"=== Simulation Results ===",
            f"Method: {self.config_summary.get('operator_type', 'N/A')}",
            f"Trials: {self.n_trials}",
            f"Total cells: {self.config_summary.get('n_total_cells', 'N/A'):,}",
            "",
            "P_top1 (cells to reach Top-1):",
            f"  Median: {p1['median']:.1f}",
            f"  Mean:   {p1['mean']:.1f} +/- {p1['std']:.1f}",
            f"  Range:  [{p1['min']:.0f}, {p1['max']:.0f}]",
            "",
            f"P_top{self.config_summary.get('topk_k', 'k')} (cells to reach Top-k):",
            f"  Median: {pk['median']:.1f}",
            f"  Mean:   {pk['mean']:.1f} +/- {pk['std']:.1f}",
            f"  Range:  [{pk['min']:.0f}, {pk['max']:.0f}]",
            "",
            "P_top100_50 (cells to reach 50 of Top-100):",
            f"  Median: {p100_50['median']:.1f}",
            f"  Mean:   {p100_50['mean']:.1f} +/- {p100_50['std']:.1f}",
            f"  Range:  [{p100_50['min']:.0f}, {p100_50['max']:.0f}]",
        ]

        return "\n".join(lines)

    def get_p_top1_values(self) -> List[int]:
        """P_top1値のリストを取得"""
        return [t.p_top1 for t in self.trials]

    def get_p_topk_values(self) -> List[int]:
        """P_topk値のリストを取得"""
        return [t.p_topk for t in self.trials]

    def get_p_top100_50_values(self) -> List[int]:
        """P_top100_50値のリストを取得"""
        return [t.p_top100_50 for t in self.trials]

    def count_initial_hits(self) -> Dict[str, int]:
        """
        初期開示でのヒット数をカウント

        Returns:
            {"top1": count, "topk": count}
        """
        return {
            "top1": sum(1 for t in self.trials if t.hit_in_initial_top1),
            "topk": sum(1 for t in self.trials if t.hit_in_initial_topk),
        }
