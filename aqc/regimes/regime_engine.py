"""
aqc/regimes/regime_engine.py
==============================
Composite Regime Detection Engine & Strategy Filter.

Orchestrates all sub-detectors into a unified ``RegimeSnapshot`` and
provides the ``RegimeFilter`` gate for regime-aware strategy switching.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from aqc.regimes.volatility_regime import VolatilityRegimeDetector, VolatilityRegime
from aqc.regimes.trend_regime import TrendRegimeDetector, TrendRegime
from aqc.regimes.correlation_regime import CorrelationRegimeDetector, CorrelationRegime
from aqc.regimes.hmm_regime import HMMRegimeDetector, HMMState

logger = logging.getLogger(__name__)


@dataclass
class RegimeSnapshot:
    """Multi-dimensional market regime at a point in time.

    Attributes
    ----------
    volatility_regime:
        Current volatility state.
    trend_regime:
        Current trend state.
    correlation_regime:
        Current correlation state.
    hmm_state:
        HMM hidden state index (-1 if unavailable).
    hmm_state_label:
        Human-readable HMM state label.
    confidence:
        Composite confidence score in [0, 1].
    timestamp:
        Snapshot time.
    """

    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    trend_regime: TrendRegime = TrendRegime.RANGE_BOUND
    correlation_regime: CorrelationRegime = CorrelationRegime.NORMAL_CORRELATION
    hmm_state: int = -1
    hmm_state_label: str = "unknown"
    confidence: float = 0.5
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary."""
        return {
            "volatility_regime": self.volatility_regime.value,
            "trend_regime": self.trend_regime.value,
            "correlation_regime": self.correlation_regime.value,
            "hmm_state": self.hmm_state,
            "hmm_state_label": self.hmm_state_label,
            "confidence": round(self.confidence, 4),
        }


class RegimeEngine:
    """Composite regime detection engine.

    Orchestrates volatility, trend, correlation, and HMM detectors
    to produce a unified ``RegimeSnapshot``.

    Parameters
    ----------
    vol_detector:
        Volatility regime detector (auto-created if None).
    trend_detector:
        Trend regime detector (auto-created if None).
    corr_detector:
        Correlation regime detector (auto-created if None).
    hmm_detector:
        HMM detector (auto-created if None).
    enable_hmm:
        Whether to run HMM fitting (default True).
    hmm_refit_every:
        Re-fit HMM every N bars (default 60).

    Examples
    --------
    >>> engine = RegimeEngine()
    >>> snapshot = engine.detect(prices_df, returns_df)
    """

    def __init__(
        self,
        vol_detector: Optional[VolatilityRegimeDetector] = None,
        trend_detector: Optional[TrendRegimeDetector] = None,
        corr_detector: Optional[CorrelationRegimeDetector] = None,
        hmm_detector: Optional[HMMRegimeDetector] = None,
        enable_hmm: bool = True,
        hmm_refit_every: int = 60,
    ) -> None:
        self.vol_detector = vol_detector or VolatilityRegimeDetector()
        self.trend_detector = trend_detector or TrendRegimeDetector()
        self.corr_detector = corr_detector or CorrelationRegimeDetector()
        self.hmm_detector = hmm_detector or HMMRegimeDetector(n_states=3)
        self.enable_hmm = enable_hmm
        self.hmm_refit_every = hmm_refit_every

        self._hmm_result: Optional[HMMState] = None
        self._call_count: int = 0

    def detect(
        self,
        prices: pd.Series,
        ohlc_df: Optional[pd.DataFrame] = None,
        multi_returns: Optional[pd.DataFrame] = None,
    ) -> RegimeSnapshot:
        """Produce a composite regime snapshot.

        Parameters
        ----------
        prices:
            Close price series for primary asset.
        ohlc_df:
            OHLC DataFrame for trend detection (must have high/low/close).
            If None, builds a synthetic one from prices.
        multi_returns:
            Multi-column returns DataFrame for correlation detection.
            If None, correlation regime defaults to NORMAL.

        Returns
        -------
        RegimeSnapshot
        """
        self._call_count += 1

        # Volatility regime
        vol_regime = self.vol_detector.detect(prices)

        # Trend regime
        if ohlc_df is not None and len(ohlc_df) > 0:
            trend_regime = self.trend_detector.detect(ohlc_df)
        else:
            # Build synthetic OHLC from close prices
            synth = pd.DataFrame({
                "open": prices, "high": prices, "low": prices, "close": prices,
            })
            trend_regime = self.trend_detector.detect(synth)

        # Correlation regime
        if multi_returns is not None and multi_returns.shape[1] >= 2:
            corr_regime = self.corr_detector.detect(multi_returns)
        else:
            corr_regime = CorrelationRegime.NORMAL_CORRELATION

        # HMM
        hmm_state = -1
        hmm_label = "disabled"
        if self.enable_hmm:
            log_ret = np.log(prices / prices.shift(1)).dropna()
            need_refit = (
                self._hmm_result is None
                or self._call_count % self.hmm_refit_every == 0
            )
            if need_refit and len(log_ret) >= 60:
                self._hmm_result = self.hmm_detector.fit(log_ret)

            if self._hmm_result is not None and self._hmm_result.state_labels is not None:
                hmm_state = int(self._hmm_result.state_labels.iloc[-1])
                hmm_label = self.hmm_detector.state_description(hmm_state)

        # Confidence score
        confidence = self._compute_confidence(vol_regime, trend_regime)

        ts = prices.index[-1] if hasattr(prices.index, "__getitem__") and len(prices) > 0 else None

        return RegimeSnapshot(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            hmm_state=hmm_state,
            hmm_state_label=hmm_label,
            confidence=confidence,
            timestamp=ts,
        )

    def detect_full_series(
        self,
        prices: pd.Series,
        ohlc_df: Optional[pd.DataFrame] = None,
        multi_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute regime labels for every bar.

        Parameters
        ----------
        prices:
            Close prices.
        ohlc_df:
            OHLC data for trend detection.
        multi_returns:
            Multi-column returns for correlation.

        Returns
        -------
        pd.DataFrame
            Columns: ``vol_regime``, ``trend_regime``, ``corr_regime``, ``hmm_state``.
        """
        # Volatility
        vol_df = self.vol_detector.detect_series(prices)

        # Trend
        if ohlc_df is not None:
            trend_df = self.trend_detector.detect_series(ohlc_df)
        else:
            synth = pd.DataFrame({
                "open": prices, "high": prices, "low": prices, "close": prices,
            })
            trend_df = self.trend_detector.detect_series(synth)

        # Correlation
        if multi_returns is not None and multi_returns.shape[1] >= 2:
            corr_df = self.corr_detector.detect_series(multi_returns)
        else:
            corr_df = pd.DataFrame(
                {"avg_corr": np.nan, "regime": CorrelationRegime.NORMAL_CORRELATION.value},
                index=prices.index,
            )

        # HMM
        log_ret = np.log(prices / prices.shift(1)).dropna()
        hmm_states = pd.Series("N/A", index=prices.index)
        if self.enable_hmm and len(log_ret) >= 60:
            try:
                result = self.hmm_detector.fit(log_ret)
                if result.state_labels is not None:
                    for idx, state in result.state_labels.items():
                        if idx in hmm_states.index:
                            hmm_states.loc[idx] = self.hmm_detector.state_description(int(state))
            except Exception as e:
                logger.debug("HMM full-series failed: %s", e)

        # Combine
        combined = pd.DataFrame(index=prices.index)
        combined["vol_regime"] = vol_df["regime"].reindex(prices.index, fill_value="NORMAL")
        combined["trend_regime"] = trend_df["regime"].reindex(prices.index, fill_value="RANGE_BOUND")
        combined["corr_regime"] = corr_df["regime"].reindex(prices.index, fill_value="NORMAL_CORRELATION")
        combined["hmm_state"] = hmm_states

        return combined

    def compute_transition_matrix(
        self, regime_series: pd.Series,
    ) -> pd.DataFrame:
        """Compute empirical state transition probabilities.

        Parameters
        ----------
        regime_series:
            Series of regime labels (strings).

        Returns
        -------
        pd.DataFrame
            Transition probability matrix (rows = from, cols = to).
        """
        states = regime_series.dropna().unique()
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(sorted(states))}

        counts = np.zeros((n, n))
        values = regime_series.dropna().values
        for i in range(1, len(values)):
            fr = state_to_idx.get(values[i - 1])
            to = state_to_idx.get(values[i])
            if fr is not None and to is not None:
                counts[fr, to] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1, None)
        probs = counts / row_sums

        labels = sorted(states)
        return pd.DataFrame(probs, index=labels, columns=labels)

    def _compute_confidence(
        self, vol_regime: VolatilityRegime, trend_regime: TrendRegime,
    ) -> float:
        """Compute a composite confidence score.

        Higher confidence when vol and trend regimes are consistent
        (e.g., low vol + range-bound, or high vol + strong trend).
        """
        consistent_pairs = {
            (VolatilityRegime.LOW, TrendRegime.RANGE_BOUND),
            (VolatilityRegime.NORMAL, TrendRegime.UPTREND),
            (VolatilityRegime.NORMAL, TrendRegime.DOWNTREND),
            (VolatilityRegime.HIGH, TrendRegime.STRONG_DOWNTREND),
            (VolatilityRegime.EXTREME, TrendRegime.STRONG_DOWNTREND),
        }

        if (vol_regime, trend_regime) in consistent_pairs:
            return 0.8
        elif vol_regime == VolatilityRegime.NORMAL:
            return 0.6
        else:
            return 0.4


# ---------------------------------------------------------------------------
# Regime Filter — Strategy/Regime Permission Gate
# ---------------------------------------------------------------------------


class RegimeFilter:
    """Decides whether a strategy should trade in the current regime.

    Parameters
    ----------
    rules:
        Mapping of ``strategy_type → set of (vol_regime, trend_regime)`` tuples
        that are *allowed*. If a strategy's type is not in the rules, it is
        always allowed.
    default_allow:
        Whether to allow trading if no rule matches (default True).

    Examples
    --------
    >>> filt = RegimeFilter(rules={
    ...     "mean_reversion": {
    ...         (VolatilityRegime.LOW, TrendRegime.RANGE_BOUND),
    ...         (VolatilityRegime.NORMAL, TrendRegime.RANGE_BOUND),
    ...     },
    ...     "momentum": {
    ...         (VolatilityRegime.NORMAL, TrendRegime.UPTREND),
    ...         (VolatilityRegime.NORMAL, TrendRegime.STRONG_UPTREND),
    ...     },
    ... })
    >>> filt.should_trade("mean_reversion", snapshot)
    True
    """

    # Pre-defined rule sets for common strategy archetypes
    DEFAULT_RULES: dict[str, set] = {
        "mean_reversion": {
            (VolatilityRegime.LOW, TrendRegime.RANGE_BOUND),
            (VolatilityRegime.NORMAL, TrendRegime.RANGE_BOUND),
            (VolatilityRegime.LOW, TrendRegime.UPTREND),
            (VolatilityRegime.LOW, TrendRegime.DOWNTREND),
            (VolatilityRegime.NORMAL, TrendRegime.UPTREND),
            (VolatilityRegime.NORMAL, TrendRegime.DOWNTREND),
        },
        "momentum": {
            (VolatilityRegime.NORMAL, TrendRegime.UPTREND),
            (VolatilityRegime.NORMAL, TrendRegime.STRONG_UPTREND),
            (VolatilityRegime.HIGH, TrendRegime.STRONG_UPTREND),
            (VolatilityRegime.NORMAL, TrendRegime.DOWNTREND),
            (VolatilityRegime.NORMAL, TrendRegime.STRONG_DOWNTREND),
            (VolatilityRegime.HIGH, TrendRegime.STRONG_DOWNTREND),
        },
    }

    def __init__(
        self,
        rules: Optional[dict[str, set]] = None,
        default_allow: bool = True,
    ) -> None:
        self.rules = rules if rules is not None else self.DEFAULT_RULES.copy()
        self.default_allow = default_allow
        self._filter_log: list[dict] = []

    def should_trade(
        self,
        strategy_type: str,
        snapshot: RegimeSnapshot,
    ) -> bool:
        """Check if a strategy should trade in the current regime.

        Parameters
        ----------
        strategy_type:
            Strategy archetype (e.g., ``"mean_reversion"``, ``"momentum"``).
        snapshot:
            Current regime snapshot.

        Returns
        -------
        bool
            True if trading is allowed.
        """
        if strategy_type not in self.rules:
            return self.default_allow

        allowed_pairs = self.rules[strategy_type]
        current = (snapshot.volatility_regime, snapshot.trend_regime)
        allowed = current in allowed_pairs

        self._filter_log.append({
            "strategy_type": strategy_type,
            "vol_regime": snapshot.volatility_regime.value,
            "trend_regime": snapshot.trend_regime.value,
            "allowed": allowed,
            "timestamp": snapshot.timestamp,
        })

        return allowed

    def add_rule(self, strategy_type: str, allowed_pairs: set) -> None:
        """Add or replace rules for a strategy type.

        Parameters
        ----------
        strategy_type:
            Strategy archetype key.
        allowed_pairs:
            Set of ``(VolatilityRegime, TrendRegime)`` tuples.
        """
        self.rules[strategy_type] = allowed_pairs

    def get_filter_log(self) -> pd.DataFrame:
        """Return the filter decision log as a DataFrame."""
        if not self._filter_log:
            return pd.DataFrame()
        return pd.DataFrame(self._filter_log)

    def activation_matrix(self) -> pd.DataFrame:
        """Compute what fraction of time each strategy was active per regime.

        Returns
        -------
        pd.DataFrame
            Rows = strategy types, columns = vol regimes, values = fraction active.
        """
        log_df = self.get_filter_log()
        if log_df.empty:
            return pd.DataFrame()

        strategies = log_df["strategy_type"].unique()
        vol_regimes = [r.value for r in VolatilityRegime]

        data = {}
        for strat in strategies:
            strat_log = log_df[log_df["strategy_type"] == strat]
            row = {}
            for vr in vol_regimes:
                mask = strat_log["vol_regime"] == vr
                if mask.sum() > 0:
                    row[vr] = round(strat_log.loc[mask, "allowed"].mean(), 4)
                else:
                    row[vr] = float("nan")
            data[strat] = row

        return pd.DataFrame(data).T
