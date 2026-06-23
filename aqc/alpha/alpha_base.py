"""
aqc/alpha/alpha_base.py
========================
Universal Alpha Interface for the AQC platform.

Every alpha signal in the system — whether driven by order book imbalance,
microprice deviation, liquidity shocks, or regime transitions — implements
this contract.  The interface is deliberately lean: four abstract methods
plus rich metadata support.

The design mirrors the existing :class:`~aqc.strategies.base_strategy.BaseStrategy`
contract (``generate_signal`` / ``on_market_event``) but operates at a higher
level of abstraction: alphas produce *prediction scores* and *evaluation
metrics* in addition to directional signals.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlphaCategory(Enum):
    """Broad classification of alpha source."""

    ORDERBOOK = "ORDERBOOK"
    MICROPRICE = "MICROPRICE"
    LIQUIDITY = "LIQUIDITY"
    ORDER_FLOW = "ORDER_FLOW"
    REGIME = "REGIME"
    ENSEMBLE = "ENSEMBLE"
    STATISTICAL = "STATISTICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    CUSTOM = "CUSTOM"


class AlphaFrequency(Enum):
    """Intended signal generation frequency."""

    TICK = "TICK"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    FIVE_MINUTE = "5MIN"
    FIFTEEN_MINUTE = "15MIN"
    HOURLY = "HOURLY"
    DAILY = "DAILY"


class AlphaStatus(Enum):
    """Lifecycle status of an alpha."""

    RESEARCH = "RESEARCH"
    BACKTESTED = "BACKTESTED"
    WALK_FORWARD = "WALK_FORWARD"
    PAPER_TRADING = "PAPER_TRADING"
    DEPLOYED = "DEPLOYED"
    RETIRED = "RETIRED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaSignal:
    """Output of a single alpha evaluation.

    Attributes
    ----------
    alpha_name:
        Identifier of the originating alpha.
    timestamp:
        Signal generation time (UTC).
    direction:
        ``1`` (long), ``-1`` (short), ``0`` (flat).
    strength:
        Normalised conviction in ``[-1.0, 1.0]``.
    confidence:
        Model confidence in ``[0.0, 1.0]``.
    features_used:
        Names of features that contributed to the signal.
    metadata:
        Arbitrary key-value pairs for audit logging.
    """

    alpha_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    direction: int = 0
    strength: float = 0.0
    confidence: float = 0.5
    features_used: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(
                f"AlphaSignal strength must be in [-1, 1], got {self.strength}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"AlphaSignal confidence must be in [0, 1], got {self.confidence}"
            )
        if self.direction not in (-1, 0, 1):
            raise ValueError(
                f"AlphaSignal direction must be -1, 0, or 1, got {self.direction}"
            )


@dataclass
class AlphaMetrics:
    """Evaluation metrics for an alpha signal.

    All metrics computed from out-of-sample predictions vs. realised returns.

    Attributes
    ----------
    sharpe_ratio:
        Annualised Sharpe of the alpha's signal PnL.
    information_coefficient:
        Rank correlation between predicted and realised returns.
    information_ratio:
        IC / std(IC) — stability of the IC.
    turnover:
        Fraction of the portfolio that changes per period.
    decay_halflife_bars:
        Number of bars until alpha signal loses half its predictive power.
    hit_rate:
        Fraction of correct directional predictions.
    profit_factor:
        Gross profit / gross loss.
    max_drawdown_pct:
        Maximum peak-to-trough decline in alpha PnL.
    calmar_ratio:
        CAGR / max drawdown.
    capacity_estimate:
        Estimated maximum capital before alpha degrades (in base currency).
    avg_holding_period:
        Average signal duration in bars.
    n_signals:
        Total number of signals generated in the evaluation period.
    """

    sharpe_ratio: float = 0.0
    information_coefficient: float = 0.0
    information_ratio: float = 0.0
    turnover: float = 0.0
    decay_halflife_bars: float = float("inf")
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    capacity_estimate: float = 0.0
    avg_holding_period: float = 0.0
    n_signals: int = 0

    def to_dict(self) -> dict[str, float]:
        """Serialise to a flat dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "information_coefficient": self.information_coefficient,
            "information_ratio": self.information_ratio,
            "turnover": self.turnover,
            "decay_halflife_bars": self.decay_halflife_bars,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "calmar_ratio": self.calmar_ratio,
            "capacity_estimate": self.capacity_estimate,
            "avg_holding_period": self.avg_holding_period,
            "n_signals": self.n_signals,
        }

    @staticmethod
    def from_predictions(
        predictions: pd.Series,
        actuals: pd.Series,
        periods_per_year: int = 252,
    ) -> "AlphaMetrics":
        """Compute metrics from aligned prediction and actual return series.

        Parameters
        ----------
        predictions:
            Alpha scores (directional predictions).
        actuals:
            Realised returns aligned to predictions.
        periods_per_year:
            Annualisation factor.

        Returns
        -------
        AlphaMetrics
        """
        aligned = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
        if len(aligned) < 5:
            logger.warning("Insufficient data for AlphaMetrics computation.")
            return AlphaMetrics()

        pred = aligned["pred"]
        actual = aligned["actual"]

        # Information Coefficient (rank correlation)
        ic = float(pred.corr(actual, method="spearman"))

        # Signal PnL: assume unit capital, long when pred > 0, short when < 0
        signal_returns = np.sign(pred) * actual
        sr_mean = signal_returns.mean()
        sr_std = signal_returns.std()

        sharpe = (
            float(sr_mean / sr_std * np.sqrt(periods_per_year))
            if sr_std > 1e-10
            else 0.0
        )

        # Hit rate
        correct = ((pred > 0) & (actual > 0)) | ((pred < 0) & (actual < 0))
        hit_rate = float(correct.mean())

        # Profit factor
        gross_profit = float(signal_returns[signal_returns > 0].sum())
        gross_loss = float(abs(signal_returns[signal_returns < 0].sum()))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-10 else 999.0
        )

        # Max drawdown on signal PnL
        cum_pnl = signal_returns.cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0

        # Turnover: fraction of bars where signal direction changes
        direction_changes = np.sign(pred).diff().abs().fillna(0)
        turnover = float(direction_changes.mean())

        # Signal count
        n_signals = int((np.sign(pred).diff().abs() > 0).sum())

        # IC stability → IR
        if len(pred) > 20:
            rolling_ic = pred.rolling(20).corr(actual)
            ic_std = rolling_ic.std()
            ir = float(ic / ic_std) if ic_std > 1e-10 else 0.0
        else:
            ir = 0.0

        # Calmar
        calmar = float(sharpe / max_dd) if max_dd > 1e-10 else 0.0

        return AlphaMetrics(
            sharpe_ratio=round(sharpe, 4),
            information_coefficient=round(ic, 4),
            information_ratio=round(ir, 4),
            turnover=round(turnover, 4),
            hit_rate=round(hit_rate, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_dd * 100, 4),
            calmar_ratio=round(calmar, 4),
            n_signals=n_signals,
        )


@dataclass
class AlphaMetadata:
    """Descriptive metadata for an alpha.

    Attributes
    ----------
    name:
        Unique identifier (e.g. ``"ob_imbalance_top5"``).
    version:
        Semantic version string.
    author:
        Creator name.
    category:
        Alpha classification.
    frequency:
        Intended signal frequency.
    status:
        Current lifecycle stage.
    universe:
        List of symbols this alpha trades.
    description:
        Human-readable summary of the alpha logic.
    parameters:
        Configuration dictionary.
    created_at:
        Creation timestamp.
    """

    name: str = "unnamed_alpha"
    version: str = "1.0.0"
    author: str = "AQC Team"
    category: AlphaCategory = AlphaCategory.CUSTOM
    frequency: AlphaFrequency = AlphaFrequency.DAILY
    status: AlphaStatus = AlphaStatus.RESEARCH
    universe: list[str] = field(default_factory=list)
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "category": self.category.value,
            "frequency": self.frequency.value,
            "status": self.status.value,
            "universe": self.universe,
            "description": self.description,
            "parameters": self.parameters,
            "created_at": str(self.created_at),
        }


# ---------------------------------------------------------------------------
# Abstract Alpha base class
# ---------------------------------------------------------------------------


class AlphaBase(ABC):
    """Universal abstract base class for all AQC alphas.

    Every alpha must implement four methods:

    - ``generate_signal`` — produce a directional signal from raw data.
    - ``predict`` — produce a continuous score for each observation.
    - ``evaluate`` — compute performance metrics.
    - ``metadata`` — return descriptive metadata.

    The base class provides:

    - Signal history tracking.
    - Automatic metric caching.
    - Convenience properties for status checks.

    Parameters
    ----------
    name:
        Unique alpha identifier.
    version:
        Semantic version string.

    Examples
    --------
    >>> class MyAlpha(AlphaBase):
    ...     def generate_signal(self, data):
    ...         return AlphaSignal(alpha_name=self.name, direction=1, strength=0.5)
    ...     def predict(self, features):
    ...         return features["score"]
    ...     def evaluate(self, predictions, actuals):
    ...         return AlphaMetrics.from_predictions(predictions, actuals)
    ...     def metadata(self):
    ...         return AlphaMetadata(name=self.name)
    """

    def __init__(self, name: str = "unnamed", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self._signal_history: list[AlphaSignal] = []
        self._cached_metrics: Optional[AlphaMetrics] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Produce a directional signal from market data.

        Parameters
        ----------
        data:
            DataFrame containing the features/bars needed by this alpha.

        Returns
        -------
        AlphaSignal
            The alpha's current view.
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Produce a continuous alpha score for each row in *features*.

        Parameters
        ----------
        features:
            Feature matrix with one row per observation.

        Returns
        -------
        pd.Series
            Predicted alpha score per observation, same index as *features*.
        """

    @abstractmethod
    def evaluate(
        self, predictions: pd.Series, actuals: pd.Series
    ) -> AlphaMetrics:
        """Compute performance metrics from predictions vs. actuals.

        Parameters
        ----------
        predictions:
            Alpha scores.
        actuals:
            Realised returns.

        Returns
        -------
        AlphaMetrics
        """

    @abstractmethod
    def metadata(self) -> AlphaMetadata:
        """Return descriptive metadata for this alpha.

        Returns
        -------
        AlphaMetadata
        """

    # ------------------------------------------------------------------
    # Optional hooks (override as needed)
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train / calibrate the alpha model.

        Override in subclasses that require a training step.

        Parameters
        ----------
        train_data:
            Training dataset.
        """
        self._is_fitted = True

    def warmup(self, data: pd.DataFrame) -> None:
        """Prime internal buffers with historical data.

        Override in alphas that maintain rolling state.

        Parameters
        ----------
        data:
            Historical data for warmup.
        """

    # ------------------------------------------------------------------
    # Signal tracking
    # ------------------------------------------------------------------

    def emit_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate, record, and return a signal.

        Wraps :meth:`generate_signal` with history tracking.

        Parameters
        ----------
        data:
            Market data input.

        Returns
        -------
        AlphaSignal
        """
        signal = self.generate_signal(data)
        self._signal_history.append(signal)
        return signal

    @property
    def signal_count(self) -> int:
        """Total signals emitted."""
        return len(self._signal_history)

    @property
    def signal_history(self) -> list[AlphaSignal]:
        """Full signal history."""
        return list(self._signal_history)

    @property
    def is_fitted(self) -> bool:
        """Whether the alpha has been trained."""
        return self._is_fitted

    @property
    def cached_metrics(self) -> Optional[AlphaMetrics]:
        """Last computed evaluation metrics."""
        return self._cached_metrics

    def cache_metrics(self, metrics: AlphaMetrics) -> None:
        """Store metrics for later retrieval."""
        self._cached_metrics = metrics

    def reset(self) -> None:
        """Clear signal history and cached metrics."""
        self._signal_history.clear()
        self._cached_metrics = None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        meta = self.metadata()
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"category={meta.category.value}, "
            f"status={meta.status.value}, "
            f"signals={self.signal_count})"
        )
