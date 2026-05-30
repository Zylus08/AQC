"""
aqc/diagnostics/trade_analyzer.py
===================================
Trade-level forensic analysis: reconstruct round-trip trades from fill log
and annotate each with regime, volatility, MFE/MAE, and signal source.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TradeRecord — one completed round-trip trade
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single round-trip trade with full forensic annotation."""

    trade_id: int = 0
    symbol: str = ""
    direction: str = ""            # LONG / SHORT
    signal_source: str = ""        # strategy_id or signal type
    entry_timestamp: Optional[pd.Timestamp] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    commission: float = 0.0
    realised_pnl: float = 0.0
    realised_return: float = 0.0   # pnl / notional
    holding_duration_days: float = 0.0
    holding_duration_bucket: str = ""   # 0-1d, 1-5d, 5-20d, 20+d
    mfe: float = 0.0              # max favourable excursion ($)
    mae: float = 0.0              # max adverse excursion ($)
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    vol_regime: str = ""
    trend_regime: str = ""
    entry_vol: float = 0.0
    exit_vol: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "signal_source": self.signal_source,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "commission": self.commission,
            "realised_pnl": self.realised_pnl,
            "realised_return": round(self.realised_return, 6),
            "holding_duration_days": round(self.holding_duration_days, 2),
            "holding_duration_bucket": self.holding_duration_bucket,
            "mfe": round(self.mfe, 2),
            "mae": round(self.mae, 2),
            "mfe_pct": round(self.mfe_pct, 4),
            "mae_pct": round(self.mae_pct, 4),
            "vol_regime": self.vol_regime,
            "trend_regime": self.trend_regime,
            "entry_vol": round(self.entry_vol, 6),
            "exit_vol": round(self.exit_vol, 6),
        }


def _duration_bucket(days: float) -> str:
    if days <= 1:
        return "0-1d"
    elif days <= 5:
        return "1-5d"
    elif days <= 20:
        return "5-20d"
    else:
        return "20+d"


# ---------------------------------------------------------------------------
# TradeAnalyzer — reconstruct trades from fill log
# ---------------------------------------------------------------------------


class TradeAnalyzer:
    """Reconstruct round-trip trades from a fill log and annotate forensically.

    Parameters
    ----------
    trade_log : list[dict]
        Fill log from ``Portfolio.trade_log``.
        Expected keys: timestamp, symbol, side, quantity, fill_price,
        commission, realised_pnl, strategy_id.
    prices : pd.DataFrame or pd.Series
        Price history for MFE/MAE calculation.
        If DataFrame, columns are symbols; if Series, single-symbol.
    regime_data : pd.DataFrame, optional
        Per-bar regime labels. Expected cols: vol_regime, trend_regime.
    vol_series : pd.Series, optional
        Per-bar volatility estimates (for annotation).
    """

    def __init__(
        self,
        trade_log: list[dict],
        prices: Optional[pd.DataFrame | pd.Series] = None,
        regime_data: Optional[pd.DataFrame] = None,
        vol_series: Optional[pd.Series] = None,
    ) -> None:
        self.trade_log = trade_log
        self.prices = prices
        self.regime_data = regime_data
        self.vol_series = vol_series
        self._trades: Optional[list[TradeRecord]] = None

    def reconstruct_trades(self) -> list[TradeRecord]:
        """Reconstruct round-trip trades from fill log.

        Pairs consecutive BUY→SELL (long trades) or SELL→BUY (short trades)
        per symbol.  Each pair produces one ``TradeRecord``.

        Returns
        -------
        list[TradeRecord]
        """
        if not self.trade_log:
            self._trades = []
            return []

        df = pd.DataFrame(self.trade_log)
        if df.empty or "symbol" not in df.columns:
            self._trades = []
            return []

        trades: list[TradeRecord] = []
        trade_id = 0

        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            pending_entry: Optional[dict] = None

            for _, row in grp.iterrows():
                side = row.get("side", "")
                if pending_entry is None:
                    # Start a new trade
                    pending_entry = row.to_dict()
                    continue

                entry_side = pending_entry.get("side", "")
                # Pair: BUY→SELL = LONG, SELL→BUY = SHORT
                is_closing = (
                    (entry_side == "BUY" and side == "SELL")
                    or (entry_side == "SELL" and side == "BUY")
                )
                if is_closing:
                    trade_id += 1
                    direction = "LONG" if entry_side == "BUY" else "SHORT"
                    entry_ts = pd.Timestamp(pending_entry["timestamp"])
                    exit_ts = pd.Timestamp(row["timestamp"])
                    entry_px = float(pending_entry.get("fill_price", 0))
                    exit_px = float(row.get("fill_price", 0))
                    qty = float(pending_entry.get("quantity", 0))
                    comm = float(pending_entry.get("commission", 0)) + float(
                        row.get("commission", 0)
                    )

                    # PnL
                    if direction == "LONG":
                        pnl = (exit_px - entry_px) * qty - comm
                    else:
                        pnl = (entry_px - exit_px) * qty - comm

                    notional = entry_px * qty if entry_px > 0 else 1.0
                    ret = pnl / notional if notional > 0 else 0.0

                    # Duration
                    dur_days = max((exit_ts - entry_ts).total_seconds() / 86400, 0)

                    # MFE / MAE
                    mfe, mae, mfe_pct, mae_pct = self._compute_excursions(
                        str(symbol), direction, entry_ts, exit_ts, entry_px, qty
                    )

                    # Regime annotation
                    vr, tr = self._get_regime(entry_ts)

                    # Volatility annotation
                    evol = self._get_vol(entry_ts)
                    xvol = self._get_vol(exit_ts)

                    trades.append(
                        TradeRecord(
                            trade_id=trade_id,
                            symbol=str(symbol),
                            direction=direction,
                            signal_source=str(
                                pending_entry.get("strategy_id", "unknown")
                            ),
                            entry_timestamp=entry_ts,
                            exit_timestamp=exit_ts,
                            entry_price=entry_px,
                            exit_price=exit_px,
                            quantity=qty,
                            commission=round(comm, 4),
                            realised_pnl=round(pnl, 2),
                            realised_return=ret,
                            holding_duration_days=dur_days,
                            holding_duration_bucket=_duration_bucket(dur_days),
                            mfe=mfe,
                            mae=mae,
                            mfe_pct=mfe_pct,
                            mae_pct=mae_pct,
                            vol_regime=vr,
                            trend_regime=tr,
                            entry_vol=evol,
                            exit_vol=xvol,
                        )
                    )
                    pending_entry = None
                else:
                    # Same-side add — update the pending entry (average in)
                    old_qty = float(pending_entry.get("quantity", 0))
                    old_px = float(pending_entry.get("fill_price", 0))
                    new_qty = float(row.get("quantity", 0))
                    new_px = float(row.get("fill_price", 0))
                    total_qty = old_qty + new_qty
                    if total_qty > 0:
                        avg_px = (old_qty * old_px + new_qty * new_px) / total_qty
                    else:
                        avg_px = new_px
                    pending_entry["quantity"] = total_qty
                    pending_entry["fill_price"] = avg_px
                    pending_entry["commission"] = float(
                        pending_entry.get("commission", 0)
                    ) + float(row.get("commission", 0))

        self._trades = trades
        logger.info("Reconstructed %d round-trip trades from %d fills",
                     len(trades), len(self.trade_log))
        return trades

    def to_dataframe(self) -> pd.DataFrame:
        """Return trades as a DataFrame."""
        trades = self._trades if self._trades is not None else self.reconstruct_trades()
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in trades])

    def top_winners(self, n: int = 10) -> pd.DataFrame:
        """Top N winning trades by PnL."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.nlargest(n, "realised_pnl").reset_index(drop=True)

    def top_losers(self, n: int = 10) -> pd.DataFrame:
        """Top N losing trades by PnL (most negative)."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.nsmallest(n, "realised_pnl").reset_index(drop=True)

    def trade_stats(self) -> dict:
        """Aggregate trade-level statistics."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        wins = df[df["realised_pnl"] > 0]
        losses = df[df["realised_pnl"] <= 0]
        return {
            "total_trades": len(df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(df), 4) if len(df) > 0 else 0,
            "avg_pnl": round(float(df["realised_pnl"].mean()), 2),
            "avg_win": round(float(wins["realised_pnl"].mean()), 2) if len(wins) > 0 else 0,
            "avg_loss": round(float(losses["realised_pnl"].mean()), 2) if len(losses) > 0 else 0,
            "best_trade": round(float(df["realised_pnl"].max()), 2),
            "worst_trade": round(float(df["realised_pnl"].min()), 2),
            "avg_duration_days": round(float(df["holding_duration_days"].mean()), 2),
            "avg_mfe_pct": round(float(df["mfe_pct"].mean()), 4),
            "avg_mae_pct": round(float(df["mae_pct"].mean()), 4),
            "profit_factor": round(
                float(wins["realised_pnl"].sum()) / abs(float(losses["realised_pnl"].sum()))
                if len(losses) > 0 and float(losses["realised_pnl"].sum()) != 0 else float("inf"),
                4,
            ),
            "total_pnl": round(float(df["realised_pnl"].sum()), 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_excursions(
        self, symbol: str, direction: str,
        entry_ts: pd.Timestamp, exit_ts: pd.Timestamp,
        entry_px: float, qty: float,
    ) -> tuple[float, float, float, float]:
        """Compute MFE and MAE for a trade."""
        if self.prices is None:
            return 0.0, 0.0, 0.0, 0.0
        try:
            if isinstance(self.prices, pd.DataFrame):
                if symbol in self.prices.columns:
                    px = self.prices[symbol].loc[entry_ts:exit_ts]
                else:
                    return 0.0, 0.0, 0.0, 0.0
            else:
                px = self.prices.loc[entry_ts:exit_ts]

            if len(px) < 2:
                return 0.0, 0.0, 0.0, 0.0

            if direction == "LONG":
                mfe_px = float(px.max()) - entry_px
                mae_px = entry_px - float(px.min())
            else:
                mfe_px = entry_px - float(px.min())
                mae_px = float(px.max()) - entry_px

            mfe = max(0, mfe_px * qty)
            mae = max(0, mae_px * qty)
            mfe_pct = mfe_px / entry_px if entry_px > 0 else 0.0
            mae_pct = mae_px / entry_px if entry_px > 0 else 0.0
            return round(mfe, 2), round(mae, 2), round(mfe_pct, 4), round(mae_pct, 4)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _get_regime(self, ts: pd.Timestamp) -> tuple[str, str]:
        """Look up regime at a timestamp."""
        if self.regime_data is None or self.regime_data.empty:
            return "", ""
        try:
            idx = self.regime_data.index.get_indexer([ts], method="ffill")[0]
            if idx < 0:
                return "", ""
            row = self.regime_data.iloc[idx]
            vr = str(row.get("vol_regime", ""))
            tr = str(row.get("trend_regime", ""))
            return vr, tr
        except Exception:
            return "", ""

    def _get_vol(self, ts: pd.Timestamp) -> float:
        """Look up volatility at a timestamp."""
        if self.vol_series is None or self.vol_series.empty:
            return 0.0
        try:
            idx = self.vol_series.index.get_indexer([ts], method="ffill")[0]
            return float(self.vol_series.iloc[idx]) if idx >= 0 else 0.0
        except Exception:
            return 0.0
