"""
Free-Wilson Operators with Discrete / Continuous Interactions

主作用 + pairwise相互作用（離散/連続）を含む探索戦略
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import BaseOperator, OperatorContext
from .registry import register_operator
from ..config import OperatorType, ContinuousInteractionModel
from ..disclosure import DisclosureState


def _double_center(table: NDArray[np.float64]) -> NDArray[np.float64]:
    """行・列平均を0にする（二重中心化）。"""
    row_mean = table.mean(axis=1, keepdims=True)
    col_mean = table.mean(axis=0, keepdims=True)
    grand_mean = table.mean()
    return table - row_mean - col_mean + grand_mean


def _predict_discrete(
    coords: NDArray[np.int64],
    intercept: float,
    main_effects: List[NDArray[np.float64]],
    interactions: Dict[Tuple[int, int], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """離散モデルの予測値を計算。"""
    n_samples = coords.shape[0]
    pred = np.full(n_samples, intercept, dtype=np.float64)
    for s, effects in enumerate(main_effects):
        pred += effects[coords[:, s]]
    for (s, t), table in interactions.items():
        pred += table[coords[:, s], coords[:, t]]
    return pred


def _fit_discrete_additive(
    coords: NDArray[np.int64],
    y: NDArray[np.float64],
    slot_sizes: Tuple[int, ...],
    alpha: float,
    max_iter: int = 5,
) -> tuple[float, List[NDArray[np.float64]], Dict[Tuple[int, int], NDArray[np.float64]]]:
    """離散相互作用の加法モデルを反復更新で学習。"""
    n_samples = len(y)
    n_slots = len(slot_sizes)

    main_effects = [
        np.zeros(size, dtype=np.float64) for size in slot_sizes
    ]
    interactions: Dict[Tuple[int, int], NDArray[np.float64]] = {}
    for s in range(n_slots):
        for t in range(s + 1, n_slots):
            interactions[(s, t)] = np.zeros((slot_sizes[s], slot_sizes[t]), dtype=np.float64)

    intercept = float(np.mean(y)) if n_samples > 0 else 0.0
    pred = _predict_discrete(coords, intercept, main_effects, interactions)

    # counts（固定）
    counts_main = [
        np.bincount(coords[:, s], minlength=slot_sizes[s]).astype(np.float64)
        for s in range(n_slots)
    ]
    counts_pair: Dict[Tuple[int, int], NDArray[np.float64]] = {}
    for (s, t), _table in interactions.items():
        counts = np.zeros((slot_sizes[s], slot_sizes[t]), dtype=np.float64)
        np.add.at(counts, (coords[:, s], coords[:, t]), 1.0)
        counts_pair[(s, t)] = counts

    for _ in range(max_iter):
        # intercept
        new_intercept = float(np.mean(y - (pred - intercept))) if n_samples > 0 else 0.0
        pred += (new_intercept - intercept)
        intercept = new_intercept

        # main effects
        for s in range(n_slots):
            old = main_effects[s]
            residual = y - pred + old[coords[:, s]]
            sum_res = np.zeros_like(old)
            np.add.at(sum_res, coords[:, s], residual)
            denom = counts_main[s] + alpha
            new = np.zeros_like(old)
            mask = denom > 0
            new[mask] = sum_res[mask] / denom[mask]
            new -= np.mean(new)
            main_effects[s] = new
            pred += (new - old)[coords[:, s]]

        # pairwise interactions
        for (s, t), old in interactions.items():
            residual = y - pred + old[coords[:, s], coords[:, t]]
            sum_res = np.zeros_like(old)
            np.add.at(sum_res, (coords[:, s], coords[:, t]), residual)
            denom = counts_pair[(s, t)] + alpha
            new = np.zeros_like(old)
            mask = denom > 0
            new[mask] = sum_res[mask] / denom[mask]
            new = _double_center(new)
            interactions[(s, t)] = new
            pred += (new - old)[coords[:, s], coords[:, t]]

    return intercept, main_effects, interactions


def _fit_bilinear_ridge(
    u_s: NDArray[np.float64],
    u_t: NDArray[np.float64],
    residual: NDArray[np.float64],
    alpha: float,
    batch_size: int = 2048,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """kron特徴量のRidge回帰でWを推定し、共分散行列の逆（H）も返す。"""
    n_samples, p = u_s.shape
    p2 = p * p
    xtx = np.zeros((p2, p2), dtype=np.float64)
    xty = np.zeros(p2, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        us = u_s[start:end]
        ut = u_t[start:end]
        r = residual[start:end]
        phi = np.einsum("bi,bj->bij", us, ut).reshape(end - start, p2)
        xtx += phi.T @ phi
        xty += phi.T @ r

    if alpha > 0:
        xtx += alpha * np.eye(p2)

    try:
        w_vec = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w_vec = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    w = w_vec.reshape(p, p)
    return w, xtx


def _predict_bilinear(
    u_s: NDArray[np.float64],
    u_t: NDArray[np.float64],
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """u_s^T W u_t を計算。"""
    return np.einsum("ni,ij,nj->n", u_s, w, u_t)


class _BaseInteractionOperator(BaseOperator):
    """相互作用を含むOperatorの共通処理"""

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self._all_pred: NDArray[np.float64] | None = None

    def _predict_all(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def update(
        self,
        new_indices: List[int],
        new_values: List[float]
    ) -> None:
        """fit()で全データ再学習するため不使用"""
        pass

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        undisclosed = disclosure_state.get_undisclosed_indices()
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []

        if self._all_pred is None:
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()

        scores = self._all_pred[undisclosed]
        return self._random_tiebreak(undisclosed, scores, actual_k)


@register_operator(OperatorType.FW_OLS_DISCRETE)
class FreeWilsonOLSDiscreteOperator(_BaseInteractionOperator):
    """Free-Wilson OLS（主作用＋離散相互作用）"""

    name = "FW-OLS-Discrete"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.intercept: float = 0.0
        self.main_effects: List[NDArray[np.float64]] | None = None
        self.interactions: Dict[Tuple[int, int], NDArray[np.float64]] | None = None

    def reset(self) -> None:
        self.intercept = 0.0
        self.main_effects = None
        self.interactions = None
        self._all_pred = None

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            return

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)
        self.intercept, self.main_effects, self.interactions = _fit_discrete_additive(
            coords, y, self.config.slot_sizes, alpha=0.0
        )
        self._all_pred = self._predict_all()

    def _predict_all(self) -> NDArray[np.float64]:
        all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
        coords = self.indexer.batch_indices_to_coords(all_indices)
        return _predict_discrete(coords, self.intercept, self.main_effects, self.interactions)


@register_operator(OperatorType.FW_RIDGE_DISCRETE)
class FreeWilsonRidgeDiscreteOperator(FreeWilsonOLSDiscreteOperator):
    """Free-Wilson Ridge（主作用＋離散相互作用）"""

    name = "FW-Ridge-Discrete"

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            return

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)
        self.intercept, self.main_effects, self.interactions = _fit_discrete_additive(
            coords, y, self.config.slot_sizes, alpha=self.config.ridge_alpha
        )
        self._all_pred = self._predict_all()


@register_operator(OperatorType.BAYESIAN_FW_DISCRETE)
class BayesianFreeWilsonDiscreteOperator(_BaseInteractionOperator):
    """ベイズFree-Wilson（主作用＋離散相互作用, UCB）"""

    name = "Bayesian-FW-Discrete"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.alpha_min = self.config.alpha_min
        self.alpha_max = self.config.alpha_max
        self.alpha = float(np.clip(self.config.ridge_alpha, self.alpha_min, self.alpha_max))
        self.beta = self.config.ucb_beta
        self.sigma_min = self.config.sigma_min
        self.intercept: float = 0.0
        self.main_effects: List[NDArray[np.float64]] | None = None
        self.interactions: Dict[Tuple[int, int], NDArray[np.float64]] | None = None
        self._all_ucb: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self.alpha = float(np.clip(self.config.ridge_alpha, self.alpha_min, self.alpha_max))
        self.intercept = 0.0
        self.main_effects = None
        self.interactions = None
        self._all_pred = None
        self._all_ucb = None

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        undisclosed = disclosure_state.get_undisclosed_indices()
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []
        if self._all_ucb is None:
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()
        scores = self._all_ucb[undisclosed]
        return self._random_tiebreak(undisclosed, scores, actual_k)

    def _update_alpha(self, theta_norm_sq: float, trace_sigma: float, n_params: int) -> None:
        if not np.isfinite(theta_norm_sq) or theta_norm_sq <= 1e-12:
            return
        gamma = n_params - self.alpha * trace_sigma
        if not np.isfinite(gamma) or gamma <= 0:
            return
        alpha_new = gamma / theta_norm_sq
        if not np.isfinite(alpha_new):
            return
        self.alpha = float(np.clip(alpha_new, self.alpha_min, self.alpha_max))

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            self._all_ucb = None
            return

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)
        self.intercept, self.main_effects, self.interactions = _fit_discrete_additive(
            coords, y, self.config.slot_sizes, alpha=self.alpha
        )

        # residual for sigma
        pred = _predict_discrete(coords, self.intercept, self.main_effects, self.interactions)
        residuals = y - pred
        if len(y) > 1:
            sigma = float(np.sqrt(np.var(residuals, ddof=1)))
        else:
            sigma = self.sigma_min
        sigma = max(sigma, self.sigma_min)

        # param variances (approx)
        counts_main = [
            np.bincount(coords[:, s], minlength=self.config.slot_sizes[s]).astype(np.float64)
            for s in range(self.config.n_slots)
        ]
        counts_pair: Dict[Tuple[int, int], NDArray[np.float64]] = {}
        for (s, t), _table in self.interactions.items():
            counts = np.zeros((self.config.slot_sizes[s], self.config.slot_sizes[t]), dtype=np.float64)
            np.add.at(counts, (coords[:, s], coords[:, t]), 1.0)
            counts_pair[(s, t)] = counts

        var_main = [
            sigma ** 2 / (counts_main[s] + self.alpha)
            for s in range(self.config.n_slots)
        ]
        var_int = {
            (s, t): sigma ** 2 / (counts_pair[(s, t)] + self.alpha)
            for (s, t) in self.interactions.keys()
        }

        # empirical bayes alpha update
        theta_norm_sq = 0.0
        trace_sigma = 0.0
        n_params = 1
        for s, effects in enumerate(self.main_effects):
            theta_norm_sq += float(np.sum(effects ** 2))
            trace_sigma += float(np.sum(var_main[s]))
            n_params += effects.size
        for (s, t), table in self.interactions.items():
            theta_norm_sq += float(np.sum(table ** 2))
            trace_sigma += float(np.sum(var_int[(s, t)]))
            n_params += table.size
        self._update_alpha(theta_norm_sq, trace_sigma, n_params)

        # predictions for all cells
        all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
        all_coords = self.indexer.batch_indices_to_coords(all_indices)
        self._all_pred = _predict_discrete(
            all_coords, self.intercept, self.main_effects, self.interactions
        )

        # sigma_param (approx, independence)
        sigma_param_sq = np.zeros_like(self._all_pred)
        for s, effects_var in enumerate(var_main):
            sigma_param_sq += effects_var[all_coords[:, s]]
        for (s, t), table_var in var_int.items():
            sigma_param_sq += table_var[all_coords[:, s], all_coords[:, t]]

        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0.0))
        self._all_ucb = self._all_pred + self.beta * sigma_param


@register_operator(OperatorType.FW_OLS_CONTINUOUS)
class FreeWilsonOLSContinuousOperator(_BaseInteractionOperator):
    """Free-Wilson OLS（主作用＋連続相互作用）"""

    name = "FW-OLS-Continuous"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.intercept: float = 0.0
        self.main_effects: List[NDArray[np.float64]] | None = None
        self.interactions: Dict[Tuple[int, int], NDArray[np.float64]] | None = None

    def reset(self) -> None:
        self.intercept = 0.0
        self.main_effects = None
        self.interactions = None
        self._all_pred = None

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            return
        if self.context.features is None:
            raise ValueError("continuous interaction requires operator features")

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)
        self.intercept, self.main_effects, self.interactions = self._fit_continuous(
            coords, y, alpha=0.0
        )
        self._all_pred = self._predict_all()

    def _fit_continuous(
        self,
        coords: NDArray[np.int64],
        y: NDArray[np.float64],
        alpha: float,
        max_iter: int = 5,
    ) -> tuple[float, List[NDArray[np.float64]], Dict[Tuple[int, int], NDArray[np.float64]]]:
        n_samples = len(y)
        n_slots = self.config.n_slots
        slot_sizes = self.config.slot_sizes
        features = self.context.features.bb_low

        main_effects = [
            np.zeros(size, dtype=np.float64) for size in slot_sizes
        ]
        interactions: Dict[Tuple[int, int], NDArray[np.float64]] = {}
        for s in range(n_slots):
            for t in range(s + 1, n_slots):
                p = features[s].shape[1]
                interactions[(s, t)] = np.zeros((p, p), dtype=np.float64)

        intercept = float(np.mean(y)) if n_samples > 0 else 0.0

        # per-sample features
        sample_features = [features[s][coords[:, s]] for s in range(n_slots)]

        # counts for main
        counts_main = [
            np.bincount(coords[:, s], minlength=slot_sizes[s]).astype(np.float64)
            for s in range(n_slots)
        ]

        pred = _predict_discrete(coords, intercept, main_effects, {
            (s, t): np.zeros((slot_sizes[s], slot_sizes[t]), dtype=np.float64)
            for s in range(n_slots) for t in range(s + 1, n_slots)
        })
        # overwrite pred with continuous interactions (currently zeros)
        pred = intercept + np.sum(
            [main_effects[s][coords[:, s]] for s in range(n_slots)], axis=0
        )

        for _ in range(max_iter):
            new_intercept = float(np.mean(y - (pred - intercept))) if n_samples > 0 else 0.0
            pred += (new_intercept - intercept)
            intercept = new_intercept

            for s in range(n_slots):
                old = main_effects[s]
                residual = y - pred + old[coords[:, s]]
                sum_res = np.zeros_like(old)
                np.add.at(sum_res, coords[:, s], residual)
                denom = counts_main[s] + alpha
                new = np.zeros_like(old)
                mask = denom > 0
                new[mask] = sum_res[mask] / denom[mask]
                new -= np.mean(new)
                main_effects[s] = new
                pred += (new - old)[coords[:, s]]

            for (s, t), old_w in interactions.items():
                u_s = sample_features[s]
                u_t = sample_features[t]
                old_contrib = _predict_bilinear(u_s, u_t, old_w)
                residual = y - pred + old_contrib
                new_w, _xtx = _fit_bilinear_ridge(u_s, u_t, residual, alpha)
                if self.config.continuous_interaction_model == ContinuousInteractionModel.LOW_RANK:
                    u, svals, vt = np.linalg.svd(new_w, full_matrices=False)
                    rank = min(self.config.continuous_interaction_rank, len(svals))
                    new_w = u[:, :rank] @ np.diag(svals[:rank]) @ vt[:rank, :]
                new_contrib = _predict_bilinear(u_s, u_t, new_w)
                pred += (new_contrib - old_contrib)
                interactions[(s, t)] = new_w

        return intercept, main_effects, interactions

    def _predict_all(self) -> NDArray[np.float64]:
        if self.context.features is None:
            raise ValueError("continuous interaction requires operator features")
        all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
        coords = self.indexer.batch_indices_to_coords(all_indices)
        pred = _predict_discrete(coords, self.intercept, self.main_effects, {
            (s, t): np.zeros((self.config.slot_sizes[s], self.config.slot_sizes[t]), dtype=np.float64)
            for s in range(self.config.n_slots) for t in range(s + 1, self.config.n_slots)
        })
        # add continuous interactions
        features = self.context.features.bb_low
        for (s, t), w in self.interactions.items():
            u_s = features[s][coords[:, s]]
            u_t = features[t][coords[:, t]]
            pred += _predict_bilinear(u_s, u_t, w)
        return pred


@register_operator(OperatorType.FW_RIDGE_CONTINUOUS)
class FreeWilsonRidgeContinuousOperator(FreeWilsonOLSContinuousOperator):
    """Free-Wilson Ridge（主作用＋連続相互作用）"""

    name = "FW-Ridge-Continuous"

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            return
        if self.context.features is None:
            raise ValueError("continuous interaction requires operator features")

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)
        self.intercept, self.main_effects, self.interactions = self._fit_continuous(
            coords, y, alpha=self.config.ridge_alpha
        )
        self._all_pred = self._predict_all()


@register_operator(OperatorType.BAYESIAN_FW_CONTINUOUS)
class BayesianFreeWilsonContinuousOperator(FreeWilsonOLSContinuousOperator):
    """ベイズFree-Wilson（主作用＋連続相互作用, UCB）"""

    name = "Bayesian-FW-Continuous"

    def __init__(self, context: OperatorContext) -> None:
        super().__init__(context)
        self.alpha_min = self.config.alpha_min
        self.alpha_max = self.config.alpha_max
        self.alpha = float(np.clip(self.config.ridge_alpha, self.alpha_min, self.alpha_max))
        self.beta = self.config.ucb_beta
        self.sigma_min = self.config.sigma_min
        self._all_ucb: NDArray[np.float64] | None = None
        self._pair_cov: Dict[Tuple[int, int], NDArray[np.float64]] = {}

    def reset(self) -> None:
        super().reset()
        self.alpha = float(np.clip(self.config.ridge_alpha, self.alpha_min, self.alpha_max))
        self._all_ucb = None
        self._pair_cov.clear()

    def select_next_cells(
        self,
        disclosure_state: DisclosureState,
        k: int
    ) -> List[int]:
        undisclosed = disclosure_state.get_undisclosed_indices()
        actual_k = min(k, len(undisclosed))
        if actual_k == 0:
            return []
        if self._all_ucb is None:
            selected = self.rng.choice(undisclosed, size=actual_k, replace=False)
            return selected.tolist()
        scores = self._all_ucb[undisclosed]
        return self._random_tiebreak(undisclosed, scores, actual_k)

    def _update_alpha(self, theta_norm_sq: float, trace_sigma: float, n_params: int) -> None:
        if not np.isfinite(theta_norm_sq) or theta_norm_sq <= 1e-12:
            return
        gamma = n_params - self.alpha * trace_sigma
        if not np.isfinite(gamma) or gamma <= 0:
            return
        alpha_new = gamma / theta_norm_sq
        if not np.isfinite(alpha_new):
            return
        self.alpha = float(np.clip(alpha_new, self.alpha_min, self.alpha_max))

    def fit(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
    ) -> None:
        if len(indices) == 0:
            self._all_pred = None
            self._all_ucb = None
            return
        if self.context.features is None:
            raise ValueError("continuous interaction requires operator features")

        coords = self.indexer.batch_indices_to_coords(indices)
        y = np.asarray(values, dtype=np.float64)

        # fit with current alpha
        self.intercept, self.main_effects, self.interactions = self._fit_continuous(
            coords, y, alpha=self.alpha
        )

        # sigma estimate
        pred = _predict_discrete(
            coords, self.intercept, self.main_effects,
            {(s, t): np.zeros((self.config.slot_sizes[s], self.config.slot_sizes[t]), dtype=np.float64)
             for s in range(self.config.n_slots) for t in range(s + 1, self.config.n_slots)}
        )
        # add continuous interactions
        features = self.context.features.bb_low
        for (s, t), w in self.interactions.items():
            pred += _predict_bilinear(features[s][coords[:, s]], features[t][coords[:, t]], w)
        residuals = y - pred
        if len(y) > 1:
            sigma = float(np.sqrt(np.var(residuals, ddof=1)))
        else:
            sigma = self.sigma_min
        sigma = max(sigma, self.sigma_min)

        # approximate covariance for parameters
        counts_main = [
            np.bincount(coords[:, s], minlength=self.config.slot_sizes[s]).astype(np.float64)
            for s in range(self.config.n_slots)
        ]
        var_main = [
            sigma ** 2 / (counts_main[s] + self.alpha)
            for s in range(self.config.n_slots)
        ]

        # per-pair covariance (diagonal only for trace)
        self._pair_cov.clear()
        trace_sigma = 0.0
        theta_norm_sq = 0.0
        n_params = 1
        for s, effects in enumerate(self.main_effects):
            theta_norm_sq += float(np.sum(effects ** 2))
            trace_sigma += float(np.sum(var_main[s]))
            n_params += effects.size

        for (s, t), w in self.interactions.items():
            u_s = features[s][coords[:, s]]
            u_t = features[t][coords[:, t]]
            _w, xtx = _fit_bilinear_ridge(u_s, u_t, residuals, self.alpha)
            # covariance ~ sigma^2 * inv(xtx)
            try:
                cov = np.linalg.inv(xtx) * (sigma ** 2)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(xtx) * (sigma ** 2)
            self._pair_cov[(s, t)] = cov
            theta_norm_sq += float(np.sum(w ** 2))
            trace_sigma += float(np.trace(cov))
            n_params += w.size

        self._update_alpha(theta_norm_sq, trace_sigma, n_params)

        # predictions for all cells
        self._all_pred = self._predict_all()

        # sigma_param for all cells (approx)
        all_indices = np.arange(self.indexer.n_total, dtype=np.int64)
        all_coords = self.indexer.batch_indices_to_coords(all_indices)
        sigma_param_sq = np.zeros_like(self._all_pred)
        for s, effects_var in enumerate(var_main):
            sigma_param_sq += effects_var[all_coords[:, s]]
        for (s, t), cov in self._pair_cov.items():
            u_s = features[s][all_coords[:, s]]
            u_t = features[t][all_coords[:, t]]
            phi = np.einsum("ni,nj->nij", u_s, u_t).reshape(len(all_coords), -1)
            sigma_param_sq += np.sum(phi @ cov * phi, axis=1)
        sigma_param = np.sqrt(np.maximum(sigma_param_sq, 0.0))
        self._all_ucb = self._all_pred + self.beta * sigma_param
