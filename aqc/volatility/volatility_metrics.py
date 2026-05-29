"""
aqc/volatility/volatility_metrics.py
======================================
Volatility-Based Position Sizing and Metrics.

Provides:

* :class:`VolatilitySizer` — Position sizing using volatility targeting.
* :func:`volatility_cone` — Rolling realized vol across multiple horizons.
* :func:`vol_of_vol` — Volatility-of-volatility metric.
* :func:`forecast_error_stats` — Forecast accuracy evaluation.

Position Sizing Models
----------------------

1. **Volatility targeting**:
   ``Position Size = (Target Risk per trade) / (Forecast Vol * Price)``

2. **Inverse volatility**:
   Weight each asset inversely proportional to its volatility.

3. **Risk parity**:
   Equal risk contribution per asset (simplified version).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methodology."""

    VOL_TARGET = "vol_target"
    INVERSE_VOL = "inverse_vol"
    RISK_PARITY = "risk_parity"


@dataclass
class SizingResult:
    """Container for position sizing output.

    Attributes
    ----------
    symbol:
        Instrument ticker.
    raw_quantity:
        Computed quantity (may be fractional).
    quantity:
        Rounded quantity.
    weight:
        Portfolio weight for this position.
    forecast_vol:
        Volatility used in sizing.
    dollar_risk:
        Dollar risk per position.
    method:
        Sizing method used.
    """

    symbol: str = ""
    raw_quantity: float = 0.0
    quantity: int = 0
    weight: float = 0.0
    forecast_vol: float = 0.0
    dollar_risk: float = 0.0
    method: SizingMethod = SizingMethod.VOL_TARGET


class VolatilitySizer:
    """Volatility-targeted position sizer.

    Sizes positions such that each position contributes approximately
    equal risk to the portfolio, based on forecast volatility.

    Parameters
    ----------
    target_vol:
        Target annualised portfolio volatility (default 0.10 = 10%).
    target_risk_per_trade:
        Target dollar risk per trade (default None — computed from equity).
    risk_fraction:
        Fraction of equity to risk per trade (default 0.01 = 1%).
    max_position_pct:
        Max position as fraction of equity (default 0.20 = 20%).
    ann_factor:
        Annualisation factor (default 252).

    Examples
    --------
    >>> sizer = VolatilitySizer(target_vol=0.10)
    >>> qty = sizer.size_position("AAPL", 150.0, 0.25, 100_000)
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        target_risk_per_trade: Optional[float] = None,
        risk_fraction: float = 0.01,
        max_position_pct: float = 0.20,
        ann_factor: int = 252,
    ) -> None:
        self.target_vol = target_vol
        self.target_risk_per_trade = target_risk_per_trade
        self.risk_fraction = risk_fraction
        self.max_position_pct = max_position_pct
        self.ann_factor = ann_factor

    def size_position(
        self,
        symbol: str,
        price: float,
        forecast_vol: float,
        equity: float,
        method: SizingMethod = SizingMethod.VOL_TARGET,
    ) -> SizingResult:
        """Compute position size based on volatility forecast.

        The core formula is:

            Position Size = Target Risk / (Daily Vol * Price)

        where ``Daily Vol = forecast_vol / sqrt(ann_factor)``.

        Parameters
        ----------
        symbol:
            Instrument ticker.
        price:
            Current price of the instrument.
        forecast_vol:
            Annualised volatility forecast (as decimal, e.g. 0.20).
        equity:
            Current portfolio equity.
        method:
            Sizing methodology (default: vol targeting).

        Returns
        -------
        SizingResult
            Position sizing details.
        """
        if price <= 0 or forecast_vol <= 0 or equity <= 0:
            return SizingResult(symbol=symbol, method=method)

        if method == SizingMethod.VOL_TARGET:
            return self._vol_target_size(symbol, price, forecast_vol, equity)
        elif method == SizingMethod.INVERSE_VOL:
            return self._inverse_vol_size(symbol, price, forecast_vol, equity)
        elif method == SizingMethod.RISK_PARITY:
            return self._risk_parity_size(symbol, price, forecast_vol, equity)
        else:
            raise ValueError(f"Unknown sizing method: {method}")

    def size_portfolio(
        self,
        symbols: list[str],
        prices: dict[str, float],
        vols: dict[str, float],
        equity: float,
        method: SizingMethod = SizingMethod.INVERSE_VOL,
    ) -> dict[str, SizingResult]:
        """Size all positions in a multi-asset portfolio.

        Parameters
        ----------
        symbols:
            List of tickers.
        prices:
            Current prices per symbol.
        vols:
            Forecast volatilities per symbol.
        equity:
            Total portfolio equity.
        method:
            Sizing method.

        Returns
        -------
        dict[str, SizingResult]
            Sizing results per symbol.
        """
        results = {}

        if method == SizingMethod.INVERSE_VOL:
            # Inverse-volatility weighting
            valid = {s: v for s, v in vols.items() if v > 0 and s in prices}
            if not valid:
                return results

            inv_vols = {s: 1.0 / v for s, v in valid.items()}
            total_inv_vol = sum(inv_vols.values())

            for s in symbols:
                if s not in valid:
                    results[s] = SizingResult(symbol=s, method=method)
                    continue

                weight = inv_vols[s] / total_inv_vol
                dollar_alloc = equity * weight
                qty_raw = dollar_alloc / prices[s]
                qty = int(qty_raw)

                results[s] = SizingResult(
                    symbol=s,
                    raw_quantity=qty_raw,
                    quantity=qty,
                    weight=round(weight, 6),
                    forecast_vol=vols[s],
                    dollar_risk=dollar_alloc * vols[s] / np.sqrt(self.ann_factor),
                    method=method,
                )

        elif method == SizingMethod.RISK_PARITY:
            # Equal risk contribution
            n = len(symbols)
            target_risk_each = equity * self.target_vol / np.sqrt(self.ann_factor) / n

            for s in symbols:
                vol = vols.get(s, 0)
                price = prices.get(s, 0)
                if vol <= 0 or price <= 0:
                    results[s] = SizingResult(symbol=s, method=method)
                    continue

                daily_vol = vol / np.sqrt(self.ann_factor)
                qty_raw = target_risk_each / (daily_vol * price)
                qty = int(qty_raw)
                weight = qty * price / equity if equity > 0 else 0

                results[s] = SizingResult(
                    symbol=s,
                    raw_quantity=qty_raw,
                    quantity=qty,
                    weight=round(weight, 6),
                    forecast_vol=vol,
                    dollar_risk=target_risk_each,
                    method=method,
                )

        else:
            # Vol target per position
            for s in symbols:
                results[s] = self.size_position(
                    s, prices.get(s, 0), vols.get(s, 0), equity, method,
                )

        return results

    # ------------------------------------------------------------------
    # Private sizing implementations
    # ------------------------------------------------------------------

    def _vol_target_size(
        self, symbol: str, price: float, forecast_vol: float, equity: float,
    ) -> SizingResult:
        """Position sizing via volatility targeting."""
        daily_vol = forecast_vol / np.sqrt(self.ann_factor)

        if self.target_risk_per_trade is not None:
            dollar_risk = self.target_risk_per_trade
        else:
            dollar_risk = equity * self.risk_fraction

        qty_raw = dollar_risk / (daily_vol * price)

        # Cap at max position
        max_qty = (equity * self.max_position_pct) / price
        qty_raw = min(qty_raw, max_qty)

        qty = int(qty_raw)
        weight = qty * price / equity if equity > 0 else 0

        return SizingResult(
            symbol=symbol,
            raw_quantity=round(qty_raw, 4),
            quantity=qty,
            weight=round(weight, 6),
            forecast_vol=forecast_vol,
            dollar_risk=round(dollar_risk, 2),
            method=SizingMethod.VOL_TARGET,
        )

    def _inverse_vol_size(
        self, symbol: str, price: float, forecast_vol: float, equity: float,
    ) -> SizingResult:
        """Single-asset inverse-vol sizing (weight = 1/vol, unnormalised)."""
        weight = self.target_vol / forecast_vol  # scale so target_vol asset gets weight=1
        weight = min(weight, self.max_position_pct * 5)  # cap

        dollar_alloc = equity * min(weight, 1.0)
        qty_raw = dollar_alloc / price
        qty = int(qty_raw)
        actual_weight = qty * price / equity if equity > 0 else 0

        return SizingResult(
            symbol=symbol,
            raw_quantity=round(qty_raw, 4),
            quantity=qty,
            weight=round(actual_weight, 6),
            forecast_vol=forecast_vol,
            dollar_risk=round(dollar_alloc * forecast_vol / np.sqrt(self.ann_factor), 2),
            method=SizingMethod.INVERSE_VOL,
        )

    def _risk_parity_size(
        self, symbol: str, price: float, forecast_vol: float, equity: float,
    ) -> SizingResult:
        """Single-asset risk-parity sizing."""
        daily_vol = forecast_vol / np.sqrt(self.ann_factor)
        target_daily_risk = equity * self.target_vol / np.sqrt(self.ann_factor)

        qty_raw = target_daily_risk / (daily_vol * price)
        max_qty = (equity * self.max_position_pct) / price
        qty_raw = min(qty_raw, max_qty)
        qty = int(qty_raw)
        weight = qty * price / equity if equity > 0 else 0

        return SizingResult(
            symbol=symbol,
            raw_quantity=round(qty_raw, 4),
            quantity=qty,
            weight=round(weight, 6),
            forecast_vol=forecast_vol,
            dollar_risk=round(target_daily_risk, 2),
            method=SizingMethod.RISK_PARITY,
        )


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------


def volatility_cone(
    prices: pd.Series,
    horizons: list[int] | None = None,
    percentiles: list[int] | None = None,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """Compute a volatility cone across multiple horizons.

    For each horizon, computes the distribution of realized volatility
    using rolling windows and returns percentile statistics.

    Parameters
    ----------
    prices:
        Close price series.
    horizons:
        List of look-back windows (default: [5, 10, 21, 63, 126, 252]).
    percentiles:
        Percentiles to compute (default: [10, 25, 50, 75, 90]).
    ann_factor:
        Annualisation factor.

    Returns
    -------
    pd.DataFrame
        Rows = horizons, columns = percentile labels + "current".
    """
    horizons = horizons or [5, 10, 21, 63, 126, 252]
    percentiles = percentiles or [10, 25, 50, 75, 90]

    log_ret = np.log(prices / prices.shift(1)).dropna()

    rows = []
    for h in horizons:
        if len(log_ret) < h:
            continue
        rolling_vol = log_ret.rolling(h).std() * np.sqrt(ann_factor)
        valid = rolling_vol.dropna()
        if valid.empty:
            continue

        row = {"horizon": h}
        for p in percentiles:
            row[f"p{p}"] = round(float(np.percentile(valid, p)), 6)
        row["current"] = round(float(valid.iloc[-1]), 6)
        rows.append(row)

    return pd.DataFrame(rows).set_index("horizon")


def vol_of_vol(
    vol_series: pd.Series,
    window: int = 21,
    ann_factor: int = 252,
) -> pd.Series:
    """Compute the volatility-of-volatility (vol-of-vol).

    Measures how unstable the volatility estimate itself is — higher
    values indicate more volatile regimes.

    Parameters
    ----------
    vol_series:
        Annualised volatility time series.
    window:
        Rolling window for computing vol-of-vol.
    ann_factor:
        Annualisation factor.

    Returns
    -------
    pd.Series
        Rolling vol-of-vol series.
    """
    vol_returns = vol_series.pct_change().dropna()
    return vol_returns.rolling(window).std() * np.sqrt(ann_factor)


def forecast_error_stats(
    forecast: pd.Series,
    realized: pd.Series,
) -> dict:
    """Compute forecast accuracy statistics.

    Parameters
    ----------
    forecast:
        Forecast volatility series.
    realized:
        Realized (ex-post) volatility series.

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``bias``, ``correlation``,
        ``hit_rate`` (% of times forecast > realized when vol rose).
    """
    # Align
    common = forecast.dropna().index.intersection(realized.dropna().index)
    if len(common) < 2:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}

    f = forecast.loc[common].values
    r = realized.loc[common].values

    errors = f - r
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    corr = float(np.corrcoef(f, r)[0, 1]) if len(f) > 1 else np.nan

    # Directional accuracy: did forecast direction match realized?
    f_change = np.diff(f)
    r_change = np.diff(r)
    if len(f_change) > 0:
        hit_rate = float(np.mean(np.sign(f_change) == np.sign(r_change)))
    else:
        hit_rate = np.nan

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "bias": round(bias, 6),
        "correlation": round(corr, 4),
        "hit_rate": round(hit_rate, 4),
        "n_observations": len(common),
    }
