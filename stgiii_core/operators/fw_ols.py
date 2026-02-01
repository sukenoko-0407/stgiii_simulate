"""
Free-Wilson OLS Operator for StageIII Simulator

古典的Free-Wilson線形モデル（OLS: 正則化なし）による探索戦略
"""

from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.FW_OLS)
class FreeWilsonOLSOperator(BaseOperator):
    """
    古典的Free-Wilson（OLS: 正則化なし）戦略

    - モデル: Free-Wilson線形モデル（スロット×BBのone-hot係数）
    - 推定: Reference codingでOLS（最小二乗法）
    - 次候補提案: 推定値μ_predの高いセルを上位からK個選択（搾取中心）

    Note:
        OLSは正則化がないため、サンプル数が特徴量数未満の段階では
        推定不可となりランダム選択にフォールバックする。
    """

    name = "FW-OLS"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.model: LinearRegression | None = None
        self._all_X: NDArray[np.float64] | None = None  # 全セルの特徴量行列（キャッシュ）

    def reset(self) -> None:
        """状態リセット"""
        self.model = None
        self._all_X = None

    def _precompute_all_features(self) -> NDArray[np.float64]:
        """全セルの特徴量行列を事前計算（予測用）"""
        if self._all_X is None:
            all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
            self._all_X = self._build_design_matrix(all_indices)
        return self._all_X

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """
        推定値μ_predの高いセルを上位からK個選択

        Args:
            disclosure_state: 現在の開示状態
            k: 選択するセル数

        Returns:
            選択したセルのインデックスリスト
        """
        undisclosed = disclosure_state.get_undisclosed_indices()
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []

        if self.model is None:
            # モデル未学習時はランダム選択
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()

        # 全セルの予測値を計算
        all_X = self._precompute_all_features()
        predictions = self.model.predict(all_X)

        # 未開示セルのみを対象に上位K個を選択
        undisclosed_scores = predictions[undisclosed]
        return self._random_tiebreak(undisclosed, undisclosed_scores, actual_k)

    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """
        新規開示データでモデルを更新

        Note:
            このメソッドは使用せず、fit()で全データを再学習する設計
        """
        pass

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64]
    ) -> None:
        """
        開示済みデータでモデルを学習（OLS: 最小二乗法）

        Args:
            indices: 開示済みセルのインデックス配列
            values: 対応する観測値配列

        Note:
            サンプル数が特徴量数未満の場合は model = None にして
            ランダムフォールバックを使用
        """
        # 空データの場合はモデルをNoneにしてランダムフォールバックを使用
        if len(indices) == 0:
            self.model = None
            return

        X = self._build_design_matrix(indices)
        y = np.asarray(values, dtype=np.float64)
        n_samples, n_features = X.shape

        # 特異行列対策: サンプル数が特徴量数未満なら推定不可
        if n_samples < n_features:
            self.model = None
            return

        # OLS（fit_intercept=False: 設計行列にinterceptを含めているため）
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X, y)

    def predict(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        指定セルの予測値を計算

        Args:
            indices: 予測対象セルのインデックス配列

        Returns:
            予測値配列
        """
        if self.model is None:
            raise ValueError("モデルが未学習です")

        X = self._build_design_matrix(indices)
        return self.model.predict(X)

    def get_coefficients_reference(self) -> NDArray[np.float64]:
        """
        Reference coding係数を取得

        Returns:
            係数配列（intercept含む）
        """
        if self.model is None:
            raise ValueError("モデルが未学習です")
        return self.model.coef_.copy()

    def get_coefficients_sum_to_zero(self) -> Dict[str, NDArray[np.float64]]:
        """
        係数をsum-to-zero表現に変換して取得

        Returns:
            各スロットの係数辞書（キー: スロット名, 値: BB係数配列）

        Note:
            Reference codingの係数をsum-to-zeroに変換
            各スロット内で係数の平均を引くことで変換
        """
        if self.model is None:
            raise ValueError("モデルが未学習です")

        coef = self.model.coef_
        result: Dict[str, NDArray[np.float64]] = {}

        col_offset = 1  # interceptをスキップ
        for slot_idx, slot_config in enumerate(self.config.slots):
            size = slot_config.n_building_blocks

            # Reference coding係数を取得（基準カテゴリは0）
            ref_coefs = np.zeros(size, dtype=np.float64)
            ref_coefs[1:] = coef[col_offset:col_offset + size - 1]

            # Sum-to-zeroに変換
            mean_coef = np.mean(ref_coefs)
            sum_to_zero_coefs = ref_coefs - mean_coef

            result[slot_config.name] = sum_to_zero_coefs
            col_offset += size - 1

        return result

    def get_intercept(self) -> float:
        """
        切片（intercept）を取得

        Returns:
            切片の値
        """
        if self.model is None:
            raise ValueError("モデルが未学習です")
        return float(self.model.coef_[0])
