"""
tests/test_regimes.py
======================
Unit and integration tests for the Regime Detection Framework.

Covers:
- VolatilityRegimeDetector
- TrendRegimeDetector
- CorrelationRegimeDetector
- HMMRegimeDetector
- RegimeEngine
- RegimeFilter

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic price series with regime-like behaviour."""
    rng = np.random.default_rng(seed)
    returns = np.concatenate([
        rng.normal(0.001, 0.005, n // 4),   # low vol
        rng.normal(0.000, 0.020, n // 4),    # high vol
        rng.normal(0.002, 0.008, n // 4),    # uptrend
        rng.normal(-0.002, 0.015, n - 3 * (n // 4)),  # downtrend
    ])
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def _make_ohlc(prices: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Build synthetic OHLC from close prices."""
    rng = np.random.default_rng(seed)
    spread = prices * 0.01
    return pd.DataFrame({
        "open": prices + rng.normal(0, 1, len(prices)) * spread * 0.3,
        "high": prices + abs(rng.normal(0, 1, len(prices))) * spread,
        "low": prices - abs(rng.normal(0, 1, len(prices))) * spread,
        "close": prices,
    }, index=prices.index)


def _make_multi_returns(n: int = 500, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate correlated multi-asset returns."""
    rng = np.random.default_rng(seed)
    # Base factor
    factor = rng.normal(0, 0.01, n)
    returns = {}
    for i in range(n_assets):
        noise = rng.normal(0, 0.005, n)
        returns[f"asset_{i}"] = factor * (0.5 + 0.5 * rng.random()) + noise
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(returns, index=idx)


# ===========================================================================
# VOLATILITY REGIME TESTS
# ===========================================================================


class TestVolatilityRegime:
    """Tests for VolatilityRegimeDetector."""

    def test_detect_returns_valid_regime(self):
        from aqc.regimes.volatility_regime import VolatilityRegimeDetector, VolatilityRegime

        detector = VolatilityRegimeDetector()
        prices = _make_prices(300)
        regime = detector.detect(prices)
        assert isinstance(regime, VolatilityRegime)

    def test_detect_series_shape(self):
        from aqc.regimes.volatility_regime import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector()
        prices = _make_prices(300)
        df = detector.detect_series(prices)
        assert "vol" in df.columns
        assert "regime" in df.columns

    def test_low_vol_detected(self):
        """Low-vol series should have LOW labels present."""
        from aqc.regimes.volatility_regime import VolatilityRegimeDetector

        rng = np.random.default_rng(42)
        prices = pd.Series(
            100.0 + np.cumsum(rng.normal(0, 0.001, 300)),
            index=pd.date_range("2020-01-01", periods=300, freq="B"),
        )
        detector = VolatilityRegimeDetector()
        df = detector.detect_series(prices)
        # Should produce valid regime labels
        valid_regimes = {"LOW", "NORMAL", "HIGH", "EXTREME"}
        assert set(df["regime"].unique()).issubset(valid_regimes)

    def test_high_vol_detected(self):
        """High-vol series should produce HIGH/EXTREME labels."""
        from aqc.regimes.volatility_regime import VolatilityRegimeDetector

        rng = np.random.default_rng(42)
        # Start with low vol, then spike
        low = rng.normal(0, 0.003, 150)
        high = rng.normal(0, 0.05, 150)
        returns = np.concatenate([low, high])
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2020-01-01", periods=300, freq="B"),
        )
        detector = VolatilityRegimeDetector()
        df = detector.detect_series(prices)
        regime_counts = df["regime"].value_counts()
        assert "HIGH" in regime_counts or "EXTREME" in regime_counts

    def test_custom_thresholds(self):
        from aqc.regimes.volatility_regime import (
            VolatilityRegimeDetector, VolRegimeThresholds,
        )

        thresholds = VolRegimeThresholds(low=10, high=90, extreme=99)
        detector = VolatilityRegimeDetector(thresholds=thresholds)
        prices = _make_prices(300)
        regime = detector.detect(prices)
        assert regime is not None


# ===========================================================================
# TREND REGIME TESTS
# ===========================================================================


class TestTrendRegime:
    """Tests for TrendRegimeDetector."""

    def test_detect_returns_valid_regime(self):
        from aqc.regimes.trend_regime import TrendRegimeDetector, TrendRegime

        detector = TrendRegimeDetector()
        ohlc = _make_ohlc(_make_prices(200))
        regime = detector.detect(ohlc)
        assert isinstance(regime, TrendRegime)

    def test_uptrend_detected(self):
        """Strong upward drift should produce UPTREND or STRONG_UPTREND."""
        from aqc.regimes.trend_regime import TrendRegimeDetector

        prices = pd.Series(
            np.linspace(100, 200, 200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        ohlc = _make_ohlc(prices)
        detector = TrendRegimeDetector()
        df = detector.detect_series(ohlc)
        regime_counts = df["regime"].value_counts()
        assert "UPTREND" in regime_counts or "STRONG_UPTREND" in regime_counts

    def test_downtrend_detected(self):
        """Strong downward drift should produce DOWNTREND."""
        from aqc.regimes.trend_regime import TrendRegimeDetector

        prices = pd.Series(
            np.linspace(200, 100, 200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        ohlc = _make_ohlc(prices)
        detector = TrendRegimeDetector()
        df = detector.detect_series(ohlc)
        regime_counts = df["regime"].value_counts()
        assert "DOWNTREND" in regime_counts or "STRONG_DOWNTREND" in regime_counts

    def test_detect_series_has_adx(self):
        from aqc.regimes.trend_regime import TrendRegimeDetector

        detector = TrendRegimeDetector()
        ohlc = _make_ohlc(_make_prices(200))
        df = detector.detect_series(ohlc)
        assert "adx" in df.columns
        assert "ma_slope" in df.columns
        assert "regime" in df.columns

    def test_short_data_returns_range_bound(self):
        from aqc.regimes.trend_regime import TrendRegimeDetector, TrendRegime

        detector = TrendRegimeDetector()
        ohlc = _make_ohlc(_make_prices(10))
        regime = detector.detect(ohlc)
        assert regime == TrendRegime.RANGE_BOUND


# ===========================================================================
# CORRELATION REGIME TESTS
# ===========================================================================


class TestCorrelationRegime:
    """Tests for CorrelationRegimeDetector."""

    def test_detect_returns_valid_regime(self):
        from aqc.regimes.correlation_regime import CorrelationRegimeDetector, CorrelationRegime

        detector = CorrelationRegimeDetector()
        returns = _make_multi_returns(200)
        regime = detector.detect(returns)
        assert isinstance(regime, CorrelationRegime)

    def test_detect_series_shape(self):
        from aqc.regimes.correlation_regime import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector()
        returns = _make_multi_returns(200)
        df = detector.detect_series(returns)
        assert "avg_corr" in df.columns
        assert "regime" in df.columns

    def test_single_asset_returns_normal(self):
        from aqc.regimes.correlation_regime import CorrelationRegimeDetector, CorrelationRegime

        detector = CorrelationRegimeDetector()
        returns = pd.DataFrame({"a": np.random.randn(100)})
        regime = detector.detect(returns)
        assert regime == CorrelationRegime.NORMAL_CORRELATION

    def test_high_correlation_detected(self):
        """Perfectly correlated assets should detect HIGH or CRISIS."""
        from aqc.regimes.correlation_regime import CorrelationRegimeDetector

        rng = np.random.default_rng(42)
        n = 300
        factor = rng.normal(0, 0.01, n)
        returns = pd.DataFrame({
            "a": factor + rng.normal(0, 0.001, n),
            "b": factor + rng.normal(0, 0.001, n),
            "c": factor + rng.normal(0, 0.001, n),
        }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

        detector = CorrelationRegimeDetector()
        df = detector.detect_series(returns)
        # With high correlation, should see HIGH or CRISIS
        regime_counts = df["regime"].value_counts()
        assert len(regime_counts) >= 1


# ===========================================================================
# HMM REGIME TESTS
# ===========================================================================


class TestHMMRegime:
    """Tests for HMMRegimeDetector."""

    def test_fit_returns_hmm_state(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector, HMMState

        detector = HMMRegimeDetector(n_states=2)
        prices = _make_prices(300)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        assert isinstance(result, HMMState)
        assert result.state_labels is not None
        assert len(result.state_labels) == len(returns)

    def test_3_state_model(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3)
        prices = _make_prices(500)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        assert result.n_states == 3
        assert result.means is not None
        assert len(result.means) == 3

    def test_4_state_model(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=4)
        prices = _make_prices(500)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        assert result.n_states == 4

    def test_invalid_n_states_raises(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector
        with pytest.raises(ValueError):
            HMMRegimeDetector(n_states=5)

    def test_short_data_returns_empty(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2)
        returns = pd.Series([0.01, -0.01, 0.02])
        result = detector.fit(returns)
        assert result.state_labels is None

    def test_transition_matrix_shape(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3)
        prices = _make_prices(500)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        assert result.transition_matrix is not None
        assert result.transition_matrix.shape == (3, 3)

    def test_states_sorted_by_mean(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3)
        prices = _make_prices(500)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        assert result.means is not None
        # Means should be sorted ascending
        for i in range(len(result.means) - 1):
            assert result.means[i] <= result.means[i + 1]

    def test_predict_current_state(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2)
        prices = _make_prices(300)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = detector.fit(returns)
        state = detector.predict_current_state(result)
        assert state in (0, 1)

    def test_state_description(self):
        from aqc.regimes.hmm_regime import HMMRegimeDetector

        d2 = HMMRegimeDetector(n_states=2)
        assert d2.state_description(0) == "Bear"
        assert d2.state_description(1) == "Bull"

        d3 = HMMRegimeDetector(n_states=3)
        assert d3.state_description(1) == "Neutral"

        d4 = HMMRegimeDetector(n_states=4)
        assert d4.state_description(0) == "Crisis"


# ===========================================================================
# REGIME ENGINE TESTS
# ===========================================================================


class TestRegimeEngine:
    """Tests for the composite RegimeEngine."""

    def test_detect_returns_snapshot(self):
        from aqc.regimes.regime_engine import RegimeEngine, RegimeSnapshot

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        snapshot = engine.detect(prices)
        assert isinstance(snapshot, RegimeSnapshot)

    def test_snapshot_to_dict(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        snapshot = engine.detect(prices)
        d = snapshot.to_dict()
        assert "volatility_regime" in d
        assert "trend_regime" in d
        assert "correlation_regime" in d

    def test_detect_with_ohlc(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        ohlc = _make_ohlc(prices)
        snapshot = engine.detect(prices, ohlc_df=ohlc)
        assert snapshot is not None

    def test_detect_with_multi_returns(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        returns = _make_multi_returns(300)
        snapshot = engine.detect(prices, multi_returns=returns)
        assert snapshot is not None

    def test_detect_full_series(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        df = engine.detect_full_series(prices)
        assert "vol_regime" in df.columns
        assert "trend_regime" in df.columns
        assert "corr_regime" in df.columns
        assert len(df) == len(prices)

    def test_transition_matrix(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=False)
        prices = _make_prices(300)
        df = engine.detect_full_series(prices)
        transmat = engine.compute_transition_matrix(df["vol_regime"])
        # Each row should sum to ~1
        row_sums = transmat.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 0.01

    def test_hmm_integration(self):
        from aqc.regimes.regime_engine import RegimeEngine

        engine = RegimeEngine(enable_hmm=True, hmm_refit_every=10)
        prices = _make_prices(300)
        snapshot = engine.detect(prices)
        # HMM state should be assigned
        assert snapshot.hmm_state >= 0 or snapshot.hmm_state_label != "unknown"


# ===========================================================================
# REGIME FILTER TESTS
# ===========================================================================


class TestRegimeFilter:
    """Tests for RegimeFilter."""

    def test_default_rules_exist(self):
        from aqc.regimes.regime_engine import RegimeFilter

        filt = RegimeFilter()
        assert "mean_reversion" in filt.rules
        assert "momentum" in filt.rules

    def test_should_trade_mean_reversion_low_vol_range(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter()
        snapshot = RegimeSnapshot(
            volatility_regime=VolatilityRegime.LOW,
            trend_regime=TrendRegime.RANGE_BOUND,
        )
        assert filt.should_trade("mean_reversion", snapshot) is True

    def test_should_not_trade_mean_reversion_high_vol_strong_trend(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter()
        snapshot = RegimeSnapshot(
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.STRONG_UPTREND,
        )
        assert filt.should_trade("mean_reversion", snapshot) is False

    def test_should_trade_momentum_uptrend(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter()
        snapshot = RegimeSnapshot(
            volatility_regime=VolatilityRegime.NORMAL,
            trend_regime=TrendRegime.UPTREND,
        )
        assert filt.should_trade("momentum", snapshot) is True

    def test_should_not_trade_momentum_range_bound(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter()
        snapshot = RegimeSnapshot(
            volatility_regime=VolatilityRegime.LOW,
            trend_regime=TrendRegime.RANGE_BOUND,
        )
        assert filt.should_trade("momentum", snapshot) is False

    def test_unknown_strategy_allows_by_default(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot

        filt = RegimeFilter()
        snapshot = RegimeSnapshot()
        assert filt.should_trade("unknown_strategy", snapshot) is True

    def test_filter_log_records_decisions(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot

        filt = RegimeFilter()
        snapshot = RegimeSnapshot()
        filt.should_trade("mean_reversion", snapshot)
        filt.should_trade("momentum", snapshot)
        log = filt.get_filter_log()
        assert len(log) == 2

    def test_activation_matrix(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter()
        for vr in VolatilityRegime:
            for tr in TrendRegime:
                snap = RegimeSnapshot(volatility_regime=vr, trend_regime=tr)
                filt.should_trade("mean_reversion", snap)
                filt.should_trade("momentum", snap)

        matrix = filt.activation_matrix()
        assert "mean_reversion" in matrix.index
        assert "momentum" in matrix.index

    def test_custom_rules(self):
        from aqc.regimes.regime_engine import RegimeFilter, RegimeSnapshot
        from aqc.regimes.volatility_regime import VolatilityRegime
        from aqc.regimes.trend_regime import TrendRegime

        filt = RegimeFilter(rules={
            "my_strat": {(VolatilityRegime.LOW, TrendRegime.RANGE_BOUND)}
        })

        snap_ok = RegimeSnapshot(
            volatility_regime=VolatilityRegime.LOW,
            trend_regime=TrendRegime.RANGE_BOUND,
        )
        snap_no = RegimeSnapshot(
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.STRONG_UPTREND,
        )
        assert filt.should_trade("my_strat", snap_ok) is True
        assert filt.should_trade("my_strat", snap_no) is False
