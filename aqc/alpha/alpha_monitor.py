"""
aqc/alpha/alpha_monitor.py
============================
Real-time alpha health monitoring.

Tracks rolling IC, signal frequency, hit rate, and detects performance
degradation using CUSUM (Cumulative Sum) change-point detection.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Alpha health classification."""

    GREEN = "GREEN"       # Performing as expected
    YELLOW = "YELLOW"     # Degradation detected
    RED = "RED"           # Significant underperformance
    CRITICAL = "CRITICAL" # Alpha should be paused


@dataclass
class HealthReport:
    """Snapshot of alpha health at a point in time.

    Attributes
    ----------
    alpha_name:
        Alpha identifier.
    timestamp:
        Report generation time.
    status:
        Overall health classification.
    rolling_ic:
        Rolling information coefficient (last window).
    rolling_hit_rate:
        Rolling directional accuracy.
    rolling_sharpe:
        Rolling annualised Sharpe of signal returns.
    signal_rate:
        Signals per period (bar).
    cusum_score:
        CUSUM statistic for change-point detection.
    cusum_triggered:
        Whether the CUSUM threshold was breached.
    alerts:
        List of human-readable alert messages.
    """

    alpha_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: HealthStatus = HealthStatus.GREEN
    rolling_ic: float = 0.0
    rolling_hit_rate: float = 0.0
    rolling_sharpe: float = 0.0
    signal_rate: float = 0.0
    cusum_score: float = 0.0
    cusum_triggered: bool = False
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alpha_name": self.alpha_name,
            "timestamp": str(self.timestamp),
            "status": self.status.value,
            "rolling_ic": round(self.rolling_ic, 4),
            "rolling_hit_rate": round(self.rolling_hit_rate, 4),
            "rolling_sharpe": round(self.rolling_sharpe, 4),
            "signal_rate": round(self.signal_rate, 4),
            "cusum_score": round(self.cusum_score, 4),
            "cusum_triggered": self.cusum_triggered,
            "alerts": self.alerts,
        }


class AlphaMonitor:
    """Monitor alpha signal quality in real time.

    Maintains rolling buffers of predictions and actuals to compute
    live health metrics.  Uses CUSUM to detect systematic performance
    shifts.

    Parameters
    ----------
    alpha_name:
        Alpha identifier for reporting.
    window:
        Rolling window for IC / hit rate / Sharpe computation.
    cusum_threshold:
        CUSUM drift threshold before triggering an alert.
    ic_warning:
        IC below this level triggers YELLOW status.
    ic_critical:
        IC below this level triggers RED status.
    ann_factor:
        Annualisation factor for Sharpe calculation.

    Examples
    --------
    >>> monitor = AlphaMonitor("my_alpha", window=50)
    >>> monitor.update(prediction=0.02, actual=0.01)
    >>> report = monitor.health_report()
    """

    def __init__(
        self,
        alpha_name: str,
        window: int = 50,
        cusum_threshold: float = 3.0,
        ic_warning: float = 0.02,
        ic_critical: float = -0.01,
        ann_factor: int = 252,
    ) -> None:
        self.alpha_name = alpha_name
        self.window = window
        self.cusum_threshold = cusum_threshold
        self.ic_warning = ic_warning
        self.ic_critical = ic_critical
        self.ann_factor = ann_factor

        self._predictions: deque[float] = deque(maxlen=window * 5)
        self._actuals: deque[float] = deque(maxlen=window * 5)
        self._signal_times: deque[datetime] = deque(maxlen=window * 5)

        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._cusum_target_ic: float = 0.05  # Expected IC in normal operation

        self._reports: list[HealthReport] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        prediction: float,
        actual: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Feed a new prediction-actual pair.

        Parameters
        ----------
        prediction:
            Alpha score / prediction.
        actual:
            Realised return.
        timestamp:
            Time of this observation.
        """
        self._predictions.append(prediction)
        self._actuals.append(actual)
        self._signal_times.append(timestamp or datetime.utcnow())

    def update_batch(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> None:
        """Feed a batch of prediction-actual pairs.

        Parameters
        ----------
        predictions:
            Alpha scores.
        actuals:
            Realised returns.
        """
        for pred, actual in zip(predictions, actuals):
            ts = None
            if hasattr(predictions, "index"):
                idx = predictions.index
                ts = idx[list(predictions).index(pred)] if len(idx) > 0 else None
            self.update(pred, actual, ts)

    # ------------------------------------------------------------------
    # Health computation
    # ------------------------------------------------------------------

    def health_report(self) -> HealthReport:
        """Generate a health report from current buffer state.

        Returns
        -------
        HealthReport
        """
        n = len(self._predictions)
        if n < 5:
            return HealthReport(
                alpha_name=self.alpha_name,
                status=HealthStatus.GREEN,
                alerts=["Insufficient data for health assessment."],
            )

        # Compute rolling metrics over last `window` observations
        win = min(self.window, n)
        preds = np.array(list(self._predictions))[-win:]
        acts = np.array(list(self._actuals))[-win:]

        # Rolling IC (rank correlation)
        pred_s = pd.Series(preds)
        act_s = pd.Series(acts)
        rolling_ic = float(pred_s.corr(act_s, method="spearman"))

        # Hit rate
        correct = ((preds > 0) & (acts > 0)) | ((preds < 0) & (acts < 0))
        rolling_hit = float(correct.mean())

        # Rolling Sharpe
        signal_rets = np.sign(preds) * acts
        sr_mean = signal_rets.mean()
        sr_std = signal_rets.std()
        rolling_sharpe = (
            float(sr_mean / sr_std * np.sqrt(self.ann_factor))
            if sr_std > 1e-10
            else 0.0
        )

        # Signal rate
        signal_rate = float(np.abs(np.sign(preds)).mean())

        # CUSUM change detection on IC stream
        ic_deviation = rolling_ic - self._cusum_target_ic
        self._cusum_pos = max(0.0, self._cusum_pos + ic_deviation)
        self._cusum_neg = min(0.0, self._cusum_neg + ic_deviation)
        cusum_score = abs(self._cusum_neg)
        cusum_triggered = cusum_score > self.cusum_threshold

        # Status classification
        alerts: list[str] = []
        if cusum_triggered:
            status = HealthStatus.CRITICAL
            alerts.append(
                f"CUSUM triggered: systematic IC degradation detected "
                f"(score={cusum_score:.2f} > threshold={self.cusum_threshold})"
            )
        elif rolling_ic < self.ic_critical:
            status = HealthStatus.RED
            alerts.append(
                f"IC below critical threshold: {rolling_ic:.4f} < {self.ic_critical}"
            )
        elif rolling_ic < self.ic_warning:
            status = HealthStatus.YELLOW
            alerts.append(
                f"IC below warning threshold: {rolling_ic:.4f} < {self.ic_warning}"
            )
        else:
            status = HealthStatus.GREEN

        if rolling_hit < 0.45:
            alerts.append(f"Hit rate below 45%: {rolling_hit:.2%}")
        if rolling_sharpe < 0.0:
            alerts.append(f"Negative rolling Sharpe: {rolling_sharpe:.2f}")

        report = HealthReport(
            alpha_name=self.alpha_name,
            status=status,
            rolling_ic=rolling_ic,
            rolling_hit_rate=rolling_hit,
            rolling_sharpe=rolling_sharpe,
            signal_rate=signal_rate,
            cusum_score=cusum_score,
            cusum_triggered=cusum_triggered,
            alerts=alerts,
        )

        self._reports.append(report)
        return report

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def report_history(self) -> list[HealthReport]:
        """All generated health reports."""
        return list(self._reports)

    def report_history_df(self) -> pd.DataFrame:
        """Health report history as a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self._reports])

    def reset_cusum(self) -> None:
        """Reset the CUSUM accumulators (e.g. after acknowledged regime change)."""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        logger.info("CUSUM reset for %s", self.alpha_name)

    @property
    def observation_count(self) -> int:
        """Number of observations ingested."""
        return len(self._predictions)
