"""
aqc/regimes/hmm_regime.py
===========================
Gaussian Hidden Markov Model Regime Detector.

Fits a Gaussian HMM to return data to discover latent market states.
Supports 2, 3, or 4-state models.

Uses ``hmmlearn`` if available; otherwise falls back to a simple
Gaussian Mixture Model (GMM) approximation via sklearn or numpy.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HMMState:
    """Container for HMM fit results.

    Attributes
    ----------
    n_states:
        Number of hidden states.
    state_labels:
        Per-bar state assignments (integer labels sorted by mean return).
    state_probs:
        Per-bar posterior probabilities for each state (n_bars x n_states).
    means:
        Mean return per state (sorted).
    variances:
        Variance per state (sorted).
    transition_matrix:
        State transition probability matrix.
    converged:
        Whether the EM algorithm converged.
    log_likelihood:
        Fitted log-likelihood.
    """

    n_states: int = 2
    state_labels: Optional[pd.Series] = None
    state_probs: Optional[np.ndarray] = None
    means: Optional[np.ndarray] = None
    variances: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    converged: bool = False
    log_likelihood: float = 0.0


class HMMRegimeDetector:
    """Gaussian HMM regime detector.

    Parameters
    ----------
    n_states:
        Number of hidden states (default 3).
    n_iter:
        Max EM iterations (default 100).
    random_state:
        Random seed for reproducibility.

    Examples
    --------
    >>> detector = HMMRegimeDetector(n_states=3)
    >>> result = detector.fit(log_returns)
    >>> current_state = result.state_labels.iloc[-1]
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if n_states not in (2, 3, 4):
            raise ValueError(f"n_states must be 2, 3, or 4, got {n_states}")
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self._has_hmmlearn = self._check_hmmlearn()

    def fit(self, returns: pd.Series) -> HMMState:
        """Fit HMM to a return series.

        Parameters
        ----------
        returns:
            Log-return or simple return series.

        Returns
        -------
        HMMState
            Fitted state assignments, means, variances, transition matrix.
        """
        r = returns.dropna()
        if len(r) < 30:
            logger.warning("HMM: insufficient data (%d bars)", len(r))
            return HMMState(n_states=self.n_states)

        if self._has_hmmlearn:
            return self._fit_hmmlearn(r)
        else:
            return self._fit_gmm_fallback(r)

    def predict_current_state(self, result: HMMState) -> int:
        """Get the most recent state label.

        Parameters
        ----------
        result:
            Fitted HMM state.

        Returns
        -------
        int
            Current state label (0 = lowest mean, n-1 = highest mean).
        """
        if result.state_labels is None:
            return 0
        return int(result.state_labels.iloc[-1])

    def state_description(self, state: int) -> str:
        """Human-readable label for a state index.

        Parameters
        ----------
        state:
            State index (0-based, sorted by mean return).

        Returns
        -------
        str
        """
        if self.n_states == 2:
            return ["Bear", "Bull"][min(state, 1)]
        elif self.n_states == 3:
            return ["Bear", "Neutral", "Bull"][min(state, 2)]
        else:
            return ["Crisis", "Bear", "Neutral", "Bull"][min(state, 3)]

    # ------------------------------------------------------------------
    # hmmlearn implementation
    # ------------------------------------------------------------------

    def _fit_hmmlearn(self, returns: pd.Series) -> HMMState:
        """Fit using hmmlearn's GaussianHMM."""
        from hmmlearn.hmm import GaussianHMM

        X = returns.values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        model.fit(X)

        # Decode most likely state sequence
        _, raw_states = model.decode(X)
        probs = model.predict_proba(X)

        # Sort states by mean return (low → high)
        means = model.means_.flatten()
        sort_idx = np.argsort(means)
        remap = {old: new for new, old in enumerate(sort_idx)}

        sorted_states = np.array([remap[s] for s in raw_states])
        sorted_means = means[sort_idx]
        sorted_vars = model.covars_.flatten()[sort_idx]
        sorted_probs = probs[:, sort_idx]

        # Reorder transition matrix
        transmat = model.transmat_[sort_idx][:, sort_idx]

        return HMMState(
            n_states=self.n_states,
            state_labels=pd.Series(sorted_states, index=returns.dropna().index, name="hmm_state"),
            state_probs=sorted_probs,
            means=sorted_means,
            variances=sorted_vars,
            transition_matrix=transmat,
            converged=model.monitor_.converged,
            log_likelihood=float(model.score(X)),
        )

    # ------------------------------------------------------------------
    # GMM fallback (no temporal structure)
    # ------------------------------------------------------------------

    def _fit_gmm_fallback(self, returns: pd.Series) -> HMMState:
        """Fallback: fit a Gaussian Mixture Model via EM (no hmmlearn).

        This is a simplified approach that clusters returns into states
        based on mean/variance but does not model state transitions.
        """
        logger.info("hmmlearn not available — using GMM fallback (no temporal structure)")

        X = returns.values
        n = len(X)
        k = self.n_states
        rng = np.random.default_rng(self.random_state)

        # K-means++ style init
        percentiles = np.linspace(10, 90, k)
        means = np.array([np.percentile(X, p) for p in percentiles])
        variances = np.full(k, float(np.var(X)))
        weights = np.full(k, 1.0 / k)

        # EM iterations
        responsibilities = np.zeros((n, k))
        converged = False
        ll_prev = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            for j in range(k):
                std = np.sqrt(max(variances[j], 1e-10))
                responsibilities[:, j] = weights[j] * (
                    1.0 / (std * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((X - means[j]) / std) ** 2)
                )

            row_sums = responsibilities.sum(axis=1, keepdims=True)
            row_sums = np.clip(row_sums, 1e-10, None)
            responsibilities /= row_sums

            # M-step
            for j in range(k):
                nk = responsibilities[:, j].sum()
                if nk < 1e-10:
                    continue
                weights[j] = nk / n
                means[j] = (responsibilities[:, j] * X).sum() / nk
                variances[j] = (responsibilities[:, j] * (X - means[j]) ** 2).sum() / nk

            # Log-likelihood
            ll = np.sum(np.log(np.clip(row_sums.flatten(), 1e-300, None)))
            if abs(ll - ll_prev) < 1e-6:
                converged = True
                break
            ll_prev = ll

        # Assign states
        raw_states = responsibilities.argmax(axis=1)

        # Sort by mean
        sort_idx = np.argsort(means)
        remap = {old: new for new, old in enumerate(sort_idx)}
        sorted_states = np.array([remap[s] for s in raw_states])
        sorted_means = means[sort_idx]
        sorted_vars = variances[sort_idx]
        sorted_probs = responsibilities[:, sort_idx]

        # Empirical transition matrix
        transmat = np.zeros((k, k))
        for i in range(1, len(sorted_states)):
            transmat[sorted_states[i - 1], sorted_states[i]] += 1
        row_sums_t = transmat.sum(axis=1, keepdims=True)
        row_sums_t = np.clip(row_sums_t, 1, None)
        transmat /= row_sums_t

        return HMMState(
            n_states=self.n_states,
            state_labels=pd.Series(sorted_states, index=returns.dropna().index, name="hmm_state"),
            state_probs=sorted_probs,
            means=sorted_means,
            variances=sorted_vars,
            transition_matrix=transmat,
            converged=converged,
            log_likelihood=ll_prev,
        )

    @staticmethod
    def _check_hmmlearn() -> bool:
        """Check if hmmlearn is installed."""
        try:
            import hmmlearn  # noqa: F401
            return True
        except ImportError:
            return False
