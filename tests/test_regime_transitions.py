"""
tests/test_regime_transitions.py
==================================
Tests for Regime Transition Alpha Research (Part 2).

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from aqc.research.regime_transitions.transition_engine import TransitionEngine, TransitionEvent
from aqc.research.regime_transitions.transition_alpha import TransitionAlphaAnalyzer


def _sample_regime_data():
    idx = pd.date_range("2020-01-01", periods=10)
    return pd.DataFrame({
        "vol_regime": ["LOW", "LOW", "NORMAL", "NORMAL", "HIGH", "HIGH", "NORMAL", "NORMAL", "LOW", "LOW"],
        "trend_regime": ["UP"] * 10
    }, index=idx)

def _sample_prices():
    idx = pd.date_range("2020-01-01", periods=10)
    return pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)


class TestTransitionEngine:
    def test_identify_transitions(self):
        engine = TransitionEngine(_sample_regime_data(), _sample_prices())
        events = engine.identify_transitions()
        
        assert len(events) == 4  # LOW->NORMAL, NORMAL->HIGH, HIGH->NORMAL, NORMAL->LOW
        assert events[0].from_regime == "LOW"
        assert events[0].to_regime == "NORMAL"
        assert events[0].price_at_transition == 102.0

    def test_get_events_df(self):
        engine = TransitionEngine(_sample_regime_data(), _sample_prices())
        df = engine.get_events_df()
        assert not df.empty
        assert "transition_pair" not in df.columns  # added later
        assert "price_at_transition" in df.columns


class TestTransitionAlpha:
    def test_compute_forward_returns(self):
        engine = TransitionEngine(_sample_regime_data(), _sample_prices())
        df = engine.get_events_df()
        
        analyzer = TransitionAlphaAnalyzer(df, _sample_prices(), horizons=[1, 3])
        ret_df = analyzer.compute_forward_returns()
        
        assert "ret_1d" in ret_df.columns
        assert "ret_3d" in ret_df.columns
        assert "transition_pair" in ret_df.columns
        assert ret_df["transition_pair"].iloc[0] == "LOW -> NORMAL"

    def test_analyze_alpha(self):
        # Need more data for t-tests
        idx = pd.date_range("2020-01-01", periods=100)
        rd = pd.DataFrame({
            "vol_regime": ["LOW"] * 25 + ["HIGH"] * 25 + ["LOW"] * 25 + ["HIGH"] * 25,
            "trend_regime": ["UP"] * 100
        }, index=idx)
        px = pd.Series(np.linspace(100, 200, 100), index=idx)
        
        engine = TransitionEngine(rd, px)
        analyzer = TransitionAlphaAnalyzer(engine.get_events_df(), px, horizons=[1])
        alpha_df = analyzer.analyze_alpha()
        
        assert not alpha_df.empty
        assert "t_stat_1d" in alpha_df.columns
        assert "p_val_1d" in alpha_df.columns

    def test_frequency_matrix(self):
        engine = TransitionEngine(_sample_regime_data(), _sample_prices())
        analyzer = TransitionAlphaAnalyzer(engine.get_events_df(), _sample_prices())
        mat = analyzer.transition_frequency_matrix("volatility")
        
        assert not mat.empty
        assert mat.loc["LOW", "NORMAL"] == 1
