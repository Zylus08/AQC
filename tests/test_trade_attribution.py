"""
tests/test_trade_attribution.py
=================================
Tests for Trade Attribution Engine (Part 1).

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import pandas as pd
import pytest

from aqc.diagnostics.trade_analyzer import TradeAnalyzer, TradeRecord
from aqc.diagnostics.trade_attribution import TradeAttributionEngine
from aqc.diagnostics.trade_reports import TradeReportGenerator

def _sample_trade_log():
    return [
        {"timestamp": pd.Timestamp("2020-01-02"), "symbol": "AAPL", "side": "BUY", "quantity": 100, "fill_price": 100.0, "commission": 1.0, "realised_pnl": 0.0, "strategy_id": "MR"},
        {"timestamp": pd.Timestamp("2020-01-05"), "symbol": "AAPL", "side": "SELL", "quantity": 100, "fill_price": 110.0, "commission": 1.0, "realised_pnl": 998.0, "strategy_id": "MR"},
        
        {"timestamp": pd.Timestamp("2020-02-01"), "symbol": "MSFT", "side": "SELL", "quantity": 50, "fill_price": 200.0, "commission": 2.0, "realised_pnl": 0.0, "strategy_id": "MOM"},
        {"timestamp": pd.Timestamp("2020-02-10"), "symbol": "MSFT", "side": "BUY", "quantity": 50, "fill_price": 210.0, "commission": 2.0, "realised_pnl": -504.0, "strategy_id": "MOM"},
    ]

def _sample_prices():
    idx = pd.date_range("2020-01-01", periods=60)
    return pd.DataFrame({
        "AAPL": [100.0 + i for i in range(60)],
        "MSFT": [200.0 + i for i in range(60)],
    }, index=idx)

def _sample_regimes():
    idx = pd.date_range("2020-01-01", periods=60)
    return pd.DataFrame({
        "vol_regime": ["NORMAL"] * 30 + ["HIGH"] * 30,
        "trend_regime": ["UPTREND"] * 60,
    }, index=idx)


class TestTradeAnalyzer:
    def test_reconstruct_long_trade(self):
        log = _sample_trade_log()[:2]
        analyzer = TradeAnalyzer(log, _sample_prices(), _sample_regimes())
        trades = analyzer.reconstruct_trades()
        assert len(trades) == 1
        t = trades[0]
        assert t.direction == "LONG"
        assert t.realised_pnl == 998.0
        assert t.holding_duration_days == 3.0
        assert t.vol_regime == "NORMAL"

    def test_reconstruct_short_trade(self):
        log = _sample_trade_log()[2:]
        analyzer = TradeAnalyzer(log, _sample_prices(), _sample_regimes())
        trades = analyzer.reconstruct_trades()
        assert len(trades) == 1
        t = trades[0]
        assert t.direction == "SHORT"
        assert t.realised_pnl == -504.0

    def test_trade_stats(self):
        analyzer = TradeAnalyzer(_sample_trade_log(), _sample_prices(), _sample_regimes())
        stats = analyzer.trade_stats()
        assert stats["total_trades"] == 2
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.5


class TestTradeAttribution:
    def test_attribution_by_regime(self):
        analyzer = TradeAnalyzer(_sample_trade_log(), _sample_prices(), _sample_regimes())
        df = analyzer.to_dataframe()
        engine = TradeAttributionEngine(df)
        
        vol_df = engine.by_vol_regime()
        assert len(vol_df) == 2  # NORMAL and HIGH
        assert "NORMAL" in vol_df["label"].values

    def test_attribution_by_signal(self):
        analyzer = TradeAnalyzer(_sample_trade_log(), _sample_prices(), _sample_regimes())
        df = analyzer.to_dataframe()
        engine = TradeAttributionEngine(df)
        
        sig_df = engine.by_signal()
        assert len(sig_df) == 2
        assert "MR" in sig_df["label"].values
        assert "MOM" in sig_df["label"].values

    def test_heatmaps(self):
        analyzer = TradeAnalyzer(_sample_trade_log(), _sample_prices(), _sample_regimes())
        df = analyzer.to_dataframe()
        engine = TradeAttributionEngine(df)
        
        hm = engine.heatmap_regime_duration()
        assert not hm.empty
        assert hm.loc["NORMAL", "1-5d"] == 998.0


class TestTradeReports:
    def test_report_generation(self):
        analyzer = TradeAnalyzer(_sample_trade_log(), _sample_prices(), _sample_regimes())
        df = analyzer.to_dataframe()
        engine = TradeAttributionEngine(df)
        
        gen = TradeReportGenerator(df, engine.full_summary(), analyzer.trade_stats())
        report = gen.build_report()
        
        assert "AQC TRADE ATTRIBUTION REPORT" in report
        assert "TRADE STATISTICS" in report
        assert "TOP 5 WINNING TRADES" in report
