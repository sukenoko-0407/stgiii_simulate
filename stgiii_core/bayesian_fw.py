"""
Bayesian Free-Wilson Operator for StageIII Simulator

ベイジアンFree-Wilson（MAP + Laplace近似 + UCB）による探索戦略
"""

from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType
from ..disclosure import DisclosureState


@register_operator(OperatorType.BAYESIAN_FW_UCB)
class BayesianFreeWilsonOperator(BaseOperator):
    """
    ベイジアンFree-Wilson（MAP + Laplace + UCB）戦略

    - モデル: 線形ガウス + 正規事前
    - 推定: MAP推定（閉形式）+ Laplace近似による係数共分散
    - 観測ノイズσ: 残差から反復推定（Empirical Bayes的）
    - 次候補提案: UCB = μ_pred + β * σ_param
    """

    name = "Bayesian-FW-UCB"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)

        # 設定パラメータ
        self.alpha_min = self.config.alpha_min
        self.alpha_max = self.config.alpha_max
        self.alpha = self._clip_alpha(self.config.ridge_alpha)  # prior precision = 1/alpha
        self.beta = self.config.ucb_beta
        self.sigma_min = self.config.sigma_min
        self.sigma_iter_max = self.config.sigma_iter_max
        self.sigma_convergence = self.config.sigma_convergence_threshold

        # 内部状態
        self.theta_map: NDArray[np.float64] | None = None  # MAP推定値
        self.Sigma_theta: NDArray[np.float64] | None = None  # 係数共分散
        self.sigma: float = self.config.sigma_gen  # 観測ノイズ推定値
        self._all_X: NDArray[np.float64] | None = None  # 全セル特徴量キャッシュ

    def reset(self) -> None:
        """状態リセット"""
        self.theta_map = None
        self.Sigma_theta = None
        self.sigma = self.config.sigma_gen
        self._all_X = None
        self.alpha = self._clip_alpha(self.config.ridge_alpha)

    def _clip_alpha(self, alpha: float) -> float:
        """alphaを安全範囲にクリップ"""
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    def _update_alpha(
        self,
        theta_map: NDArray[np.float64],
        Sigma_theta: NDArray[np.float64]
    ) -> float:
        """
        Evidence Maximization (ML-II) によるalpha更新

        更新式:
            γ = n_features - α * trace(Σ)
            α_new = γ / (θ^T θ)
        """
        theta_norm_sq = float(theta_map.T @ theta_map)
        if not np.isfinite(theta_norm_sq) or theta_norm_sq <= 1e-12:
            return self.alpha

        trace_sigma = float(np.trace(Sigma_theta))
        gamma = Sigma_theta.shape[0] - self.alpha * trace_sigma
        if not np.isfinite(gamma) or gamma <= 0:
            return self.alpha

        alpha_new = gamma / theta_norm_sq
        if not np.isfinite(alpha_new):
            return self.alpha

        return self._clip_alpha(alpha_new)

    def _precompute_all_features(self) -> NDArray[np.float64]:
        """全セルの特徴量行列を事前計算"""
        if self._all_X is None:
            all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
            self._all_X = self._build_design_matrix(all_indices)
        return self._all_X

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64]
    ) -> None:
        """
        開示済みデータでMAP推定 + Laplace近似

        Args:
            indices: 開示済みセルのインデックス配列
            values: 対応する観測値配列

        Note:
            σの反復推定を行い、MAP推定値と係数共分散を計算
        """
        # 空データの場合はモデルをNoneにしてランダムフォールバックを使用
        if len(indices) == 0:
            self.theta_map = None
            self.Sigma_theta = None
            return

        X = self._build_design_matrix(indices)
        y = np.asarray(values, dtype=np.float64)
        n_samples, n_features = X.shape

        # Prior precision matrix: Lambda^{-1} = alpha * I
        Lambda_inv = self.alpha * np.eye(n_features)

        # X'X（一度だけ計算）
        XtX = X.T @ X
        Xty = X.T @ y

        # σ推定の反復
        sigma = self.sigma
        theta = None

        for iteration in range(self.sigma_iter_max):
            sigma_old = sigma

            # ヘッセ行列: H = X'X/sigma^2 + Lambda^{-1}
            H = XtX / (sigma ** 2) + Lambda_inv

            # MAP推定: theta = H^{-1} * X'y / sigma^2
            try:
                c, lower = cho_factor(H)
                theta = cho_solve((c, lower), Xty / (sigma ** 2))
            except np.linalg.LinAlgError:
                # コレスキー分解失敗時は通常の逆行列
                theta = np.linalg.solve(H, Xty / (sigma ** 2))

            # 残差からσを更新
            residuals = y - X @ theta
            if n_samples > 1:
                sigma_hat = np.sqrt(np.var(residuals, ddof=1))
            else:
                sigma_hat = self.sigma_min

            sigma = max(sigma_hat, self.sigma_min)

            # 収束判定
            if abs(sigma - sigma_old) / max(sigma_old, 1e-10) < self.sigma_convergence:
                break

        self.sigma = sigma
        self.theta_map = theta

        # Laplace近似による係数共分散: Sigma_theta = H^{-1}
        H = XtX / (self.sigma ** 2) + Lambda_inv
        try:
            c, lower = cho_factor(H)
            self.Sigma_theta = cho_solve((c, lower), np.eye(n_features))
        except np.linalg.LinAlgError:
            self.Sigma_theta = np.linalg.inv(H)

        # Empirical Bayes: alpha更新（fit毎に1回）
        new_alpha = self._update_alpha(self.theta_map, self.Sigma_theta)
        if abs(new_alpha - self.alpha) > 1e-12:
            self.alpha = new_alpha
            Lambda_inv = self.alpha * np.eye(n_features)
            H = XtX / (self.sigma ** 2) + Lambda_inv
            try:
                c, lower = cho_factor(H)
                self.theta_map = cho_solve((c, lower), Xty / (self.sigma ** 2))
                self.Sigma_theta = cho_solve((c, lower), np.eye(n_features))
            except np.linalg.LinAlgError:
                self.theta_map = np.linalg.solve(H, Xty / (self.sigma ** 2))
                self.Sigma_theta = np.linalg.inv(H)

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """
        UCBスコアの高いセルを上位からK個選択

        Args:
            disclosure_state: 現在の開示状態
            k: 選択するセル数

        Returns:
            選択したセルのインデックスリスト

        Note:
            UCB = μ_pred + β * σ_param
            σ_paramは係数事後由来の不確実性のみ（観測ノイズ含まず）
        """
        undisclosed = disclosure_state.get_undisclosed_indices()
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []

        if self.theta_map is None:
            # モデル未学習時はランダム選択
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()

        # 全セルの予測値と不確実性を計算
        all_X = self._precompute_all_features()
        mu_pred = all_X @ self.theta_map

        # 係数由来の不確実性: sigma_param^2 = x' Sigma_theta x
        # 効率化: (X @ Sigma_theta) * X の行方向和
        XS = all_X @ self.Sigma_theta
        sigma_param_sq = np.sum(XS * all_X, axis=1)
        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0))

        # UCBスコア
        ucb_scores = mu_pred + self.beta * sigma_param

        # 未開示セルのみを対象に上位K個を選択
        undisclosed_scores = ucb_scores[undisclosed]
        return self._random_tiebreak(undisclosed, undisclosed_scores, actual_k)

    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """新規開示データでモデルを更新（fit()で全データ再計算するため不使用）"""
        pass

    def predict(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        指定セルの予測平均値を計算

        Args:
            indices: 予測対象セルのインデックス配列

        Returns:
            予測平均値配列
        """
        if self.theta_map is None:
            raise ValueError("モデルが未学習です")

        X = self._build_design_matrix(indices)
        return X @ self.theta_map

    def predict_with_uncertainty(
        self,
        indices: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        予測値と不確実性を計算

        Args:
            indices: 予測対象セルのインデックス配列

        Returns:
            (mu_pred, sigma_param, sigma_total) のタプル
            - mu_pred: 予測平均
            - sigma_param: 係数由来の不確実性（探索項用）
            - sigma_total: 総不確実性（予測区間用）
        """
        if self.theta_map is None or self.Sigma_theta is None:
            raise ValueError("モデルが未学習です")

        X = self._build_design_matrix(indices)
        mu_pred = X @ self.theta_map

        # 係数由来の不確実性
        XS = X @ self.Sigma_theta
        sigma_param_sq = np.sum(XS * X, axis=1)
        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0))

        # 総不確実性（係数由来 + 観測ノイズ）
        sigma_total = np.sqrt(sigma_param_sq + self.sigma ** 2)

        return mu_pred, sigma_param, sigma_total

    def _sample_theta(self) -> NDArray[np.float64]:
        """係数事後からサンプリング"""
        if self.theta_map is None or self.Sigma_theta is None:
            raise ValueError("モデルが未学習です")

        n_features = self.theta_map.shape[0]
        try:
            L = np.linalg.cholesky(self.Sigma_theta)
            z = self.rng.standard_normal(n_features)
            return self.theta_map + L @ z
        except np.linalg.LinAlgError:
            # 数値誤差で非正定値になる場合のフォールバック
            eigvals, eigvecs = np.linalg.eigh(self.Sigma_theta)
            eigvals = np.clip(eigvals, 0.0, None)
            z = self.rng.standard_normal(n_features)
            return self.theta_map + eigvecs @ (np.sqrt(eigvals) * z)

    def get_coefficients_reference(self) -> NDArray[np.float64]:
        """Reference coding係数を取得"""
        if self.theta_map is None:
            raise ValueError("モデルが未学習です")
        return self.theta_map.copy()

    def get_coefficients_sum_to_zero(self) -> Dict[str, NDArray[np.float64]]:
        """
        係数をsum-to-zero表現に変換して取得

        Returns:
            各スロットの係数辞書（キー: スロット名, 値: BB係数配列）
        """
        if self.theta_map is None:
            raise ValueError("モデルが未学習です")

        coef = self.theta_map
        result: Dict[str, NDArray[np.float64]] = {}

        col_offset = 1  # interceptをスキップ
        for slot_idx, slot_config in enumerate(self.config.slots):
            size = slot_config.n_building_blocks

            # Reference coding係数（基準カテゴリ=0の係数は0）
            ref_coefs = np.zeros(size, dtype=np.float64)
            ref_coefs[1:] = coef[col_offset:col_offset + size - 1]

            # Sum-to-zeroに変換
            mean_coef = np.mean(ref_coefs)
            sum_to_zero_coefs = ref_coefs - mean_coef

            result[slot_config.name] = sum_to_zero_coefs
            col_offset += size - 1

        return result

    def get_intercept(self) -> float:
        """切片を取得"""
        if self.theta_map is None:
            raise ValueError("モデルが未学習です")
        return float(self.theta_map[0])

    def get_sigma(self) -> float:
        """推定された観測ノイズσを取得"""
        return self.sigma

    def get_coefficient_covariance(self) -> NDArray[np.float64]:
        """係数共分散行列を取得"""
        if self.Sigma_theta is None:
            raise ValueError("モデルが未学習です")
        return self.Sigma_theta.copy()


@register_operator(OperatorType.BAYESIAN_FW_TS)
class BayesianFreeWilsonTSOperator(BayesianFreeWilsonOperator):
    """
    ベイジアンFree-Wilson（MAP + Laplace + Thompson Sampling）戦略
    """

    name = "Bayesian-FW-TS"

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        """
        Thompson Samplingで次候補を選択

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

        if self.theta_map is None or self.Sigma_theta is None:
            # モデル未学習時はランダム選択
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()

        all_X = self._precompute_all_features()
        theta_sample = self._sample_theta()
        scores = all_X @ theta_sample

        undisclosed_scores = scores[undisclosed]
        return self._random_tiebreak(undisclosed, undisclosed_scores, actual_k)
