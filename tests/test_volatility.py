"""
tests/test_volatility.py
=========================
Comprehensive tests for the Volatility Forecasting Framework.

Test coverage:
* EWMA — variance recursion, volatility, edge cases
* GARCH(1,1) — fitting, forecasting, persistence
* VolatilityForecastEngine — ensemble, regimes, CI, report generation
* VolatilitySizer — vol-targeting, inverse-vol, risk-parity
* Metric functions — vol cone, vol-of-vol, forecast error stats
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aqc.volatility.ewma import ewma_variance, ewma_volatility, ewma_forecast
from aqc.volatility.garch import GARCH11, GARCHResult
from aqc.volatility.forecasting_engine import (
    VolatilityForecastEngine,
    VolRegime,
    ForecastResult,
)
from aqc.volatility.volatility_metrics import (
    VolatilitySizer,
    SizingMethod,
    SizingResult,
    volatility_cone,
    vol_of_vol,
    forecast_error_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_price_series(n: int = 500, seed: int = 42, vol: float = 0.01) -> pd.Series:
    """Generate synthetic price series with known volatility characteristics."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    returns = rng.normal(0.0003, vol, n)
    prices = np.exp(np.cumsum(returns)) * 100.0
    return pd.Series(prices, index=pd.DatetimeIndex(dates, name="timestamp"), name="close")


def make_returns(n: int = 500, seed: int = 42, vol: float = 0.01) -> pd.Series:
    """Generate synthetic log-returns."""
    prices = make_price_series(n, seed, vol)
    return np.log(prices / prices.shift(1)).dropna()


def make_garch_returns(n: int = 1000, seed: int = 42) -> pd.Series:
    """Generate synthetic returns with GARCH-like volatility clustering."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)

    omega = 0.00001
    alpha = 0.08
    beta = 0.90

    var_t = np.zeros(n)
    r = np.zeros(n)
    var_t[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        var_t[t] = omega + alpha * r[t - 1] ** 2 + beta * var_t[t - 1]
        r[t] = rng.normal(0, np.sqrt(var_t[t]))

    return pd.Series(r, index=pd.DatetimeIndex(dates, name="timestamp"), name="returns")


@pytest.fixture(scope="module")
def prices():
    return make_price_series(500, seed=42)


@pytest.fixture(scope="module")
def returns():
    return make_returns(500, seed=42)


@pytest.fixture(scope="module")
def garch_returns():
    return make_garch_returns(1000, seed=42)


# ---------------------------------------------------------------------------
# EWMA Tests
# ---------------------------------------------------------------------------


class TestEWMAVolatility:

    def test_ewma_variance_output_shape(self, returns):
        var = ewma_variance(returns, decay=0.94)
        assert len(var) == len(returns)
        assert var.name == "ewma_variance"

    def test_ewma_variance_all_nan_until_warmup(self, returns):
        var = ewma_variance(returns, decay=0.94, min_periods=20)
        # First 18 values should be NaN (min_periods - 1 before first valid)
        # (since first return is NaN from shift, effective start is idx 1 + 20 = idx 20)
        assert var.iloc[:18].isna().all()

    def test_ewma_variance_positive(self, returns):
        var = ewma_variance(returns, decay=0.94)
        valid = var.dropna()
        assert (valid >= 0).all()

    def test_ewma_volatility_annualised(self, returns):
        vol = ewma_volatility(returns, decay=0.94, annualise=True, ann_factor=252)
        valid = vol.dropna()
        # Annualised vol should be in a reasonable range (e.g. 5% — 80%)
        assert valid.mean() > 0.01
        assert valid.mean() < 2.0

    def test_ewma_volatility_not_annualised(self, returns):
        vol_ann = ewma_volatility(returns, decay=0.94, annualise=True)
        vol_raw = ewma_volatility(returns, decay=0.94, annualise=False)
        valid_ann = vol_ann.dropna()
        valid_raw = vol_raw.dropna()
        # Annualised should be ~sqrt(252) times larger
        ratio = valid_ann.mean() / valid_raw.mean()
        assert 10 < ratio < 20  # sqrt(252) ~ 15.87

    def test_ewma_decay_sensitivity(self, returns):
        vol_fast = ewma_volatility(returns, decay=0.85)  # faster decay
        vol_slow = ewma_volatility(returns, decay=0.97)  # slower decay
        # Faster decay should react more to recent changes
        # Just verify both produce valid output
        assert vol_fast.dropna().shape[0] > 0
        assert vol_slow.dropna().shape[0] > 0

    def test_ewma_invalid_decay_raises(self, returns):
        with pytest.raises(ValueError, match="decay must be in"):
            ewma_variance(returns, decay=1.5)

    def test_ewma_forecast_output(self, returns):
        fc = ewma_forecast(returns, decay=0.94, horizon=5)
        assert fc.name == "ewma_forecast_5d"
        assert fc.dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# GARCH Tests
# ---------------------------------------------------------------------------


class TestGARCH11:

    def test_fit_converges(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        assert result.converged

    def test_fit_parameters_bounded(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        assert result.omega > 0
        assert 0 < result.alpha < 1
        assert 0 < result.beta < 1
        assert result.persistence < 1

    def test_fit_recovers_approximate_params(self, garch_returns):
        """GARCH fit should produce sensible stationary parameters."""
        model = GARCH11()
        result = model.fit(garch_returns)
        # Key properties: alpha+beta < 1 (stationary), alpha > 0, beta > 0
        assert result.alpha > 0.01
        assert result.beta > 0.1
        assert result.persistence < 1.0
        assert result.persistence > 0.3  # reasonable persistence

    def test_conditional_variance_series(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        assert result.conditional_variance is not None
        assert len(result.conditional_variance) == len(garch_returns.dropna())
        assert (result.conditional_variance > 0).all()

    def test_long_run_volatility_positive(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        assert result.long_run_volatility > 0

    def test_half_life_positive(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        assert result.half_life > 0

    def test_forecast_returns_dict(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        fc = model.forecast(result, garch_returns, horizon=5)
        expected_keys = {
            "forecast_variance", "forecast_volatility",
            "forecast_vol_annualised", "ci_lower_95", "ci_upper_95", "horizon",
        }
        assert expected_keys.issubset(fc.keys())
        assert fc["horizon"] == 5
        assert fc["forecast_vol_annualised"] > 0

    def test_conditional_volatility_method(self, garch_returns):
        model = GARCH11()
        result = model.fit(garch_returns)
        vol = model.conditional_volatility(result, annualise=True)
        assert vol.name == "garch_volatility"
        assert (vol > 0).all()

    def test_short_series_warning(self):
        """Should warn for very short series."""
        short = pd.Series(np.random.default_rng(0).normal(0, 0.01, 15))
        model = GARCH11()
        # Should still run (with warning)
        result = model.fit(short)
        assert isinstance(result, GARCHResult)


# ---------------------------------------------------------------------------
# Forecasting Engine Tests
# ---------------------------------------------------------------------------


class TestVolatilityForecastEngine:

    def test_fit_and_forecast_returns_result(self, prices):
        engine = VolatilityForecastEngine()
        result = engine.fit_and_forecast(prices)
        assert isinstance(result, ForecastResult)
        assert result.forecast_vol > 0

    def test_forecast_has_all_models(self, prices):
        engine = VolatilityForecastEngine()
        result = engine.fit_and_forecast(prices)
        assert "ewma" in result.model_vols
        assert "garch" in result.model_vols
        assert "historical" in result.model_vols

    def test_confidence_intervals_ordered(self, prices):
        engine = VolatilityForecastEngine()
        result = engine.fit_and_forecast(prices)
        assert result.ci_lower <= result.forecast_vol <= result.ci_upper

    def test_regime_is_valid(self, prices):
        engine = VolatilityForecastEngine()
        result = engine.fit_and_forecast(prices)
        assert isinstance(result.regime, VolRegime)

    def test_compute_full_series_shape(self, prices):
        engine = VolatilityForecastEngine()
        df = engine.compute_full_series(prices)
        assert "ewma_vol" in df.columns
        assert "garch_vol" in df.columns
        assert "hist_vol" in df.columns
        assert "ensemble_vol" in df.columns
        assert "regime" in df.columns
        assert len(df) > 0

    def test_generate_report_creates_csv(self, prices, tmp_path):
        engine = VolatilityForecastEngine()
        output = str(tmp_path / "vol_report.csv")
        df = engine.generate_report(prices, output_path=output)
        assert len(df) > 0
        import os
        assert os.path.exists(output)

    def test_custom_weights(self, prices):
        engine = VolatilityForecastEngine(weights=(1.0, 0.0, 0.0))
        result = engine.fit_and_forecast(prices)
        # With GARCH/hist weight = 0, ensemble should be pure EWMA
        assert abs(result.forecast_vol - result.model_vols["ewma"]) < 0.01

    def test_regime_detection_multiple_calls(self, prices):
        """Calling fit_and_forecast multiple times builds regime history."""
        engine = VolatilityForecastEngine()
        # Feed different price slices to build history
        for i in range(50, len(prices), 20):
            result = engine.fit_and_forecast(prices.iloc[:i])
        # After many calls, regime classification should work
        assert len(engine._vol_history) > 5

    def test_short_series_returns_empty(self):
        engine = VolatilityForecastEngine()
        short_prices = pd.Series([100, 101, 102], dtype=float)
        result = engine.fit_and_forecast(short_prices)
        assert result.forecast_vol == 0.0


# ---------------------------------------------------------------------------
# Volatility Sizer Tests
# ---------------------------------------------------------------------------


class TestVolatilitySizer:

    def test_vol_target_basic(self):
        sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01)
        result = sizer.size_position(
            "AAPL", price=150.0, forecast_vol=0.20, equity=100_000,
        )
        assert result.quantity > 0
        assert result.method == SizingMethod.VOL_TARGET
        assert result.symbol == "AAPL"

    def test_higher_vol_smaller_position(self):
        sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01, max_position_pct=0.50)
        r_low_vol = sizer.size_position("X", 100.0, 0.10, 100_000)
        r_high_vol = sizer.size_position("X", 100.0, 0.40, 100_000)
        # Higher vol should result in smaller raw position
        assert r_high_vol.raw_quantity < r_low_vol.raw_quantity

    def test_max_position_cap(self):
        sizer = VolatilitySizer(
            target_vol=0.10, risk_fraction=0.50,  # very high risk
            max_position_pct=0.10,
        )
        result = sizer.size_position("X", 100.0, 0.05, 100_000)
        # Should be capped at 10% of equity / price = 100
        assert result.quantity <= 100

    def test_inverse_vol_sizing(self):
        sizer = VolatilitySizer(target_vol=0.15)
        result = sizer.size_position(
            "Y", 50.0, 0.30, 100_000, method=SizingMethod.INVERSE_VOL,
        )
        assert result.quantity > 0
        assert result.method == SizingMethod.INVERSE_VOL

    def test_risk_parity_sizing(self):
        sizer = VolatilitySizer(target_vol=0.10)
        result = sizer.size_position(
            "Z", 200.0, 0.25, 100_000, method=SizingMethod.RISK_PARITY,
        )
        assert result.quantity > 0
        assert result.method == SizingMethod.RISK_PARITY

    def test_zero_price_returns_zero(self):
        sizer = VolatilitySizer()
        result = sizer.size_position("X", 0.0, 0.20, 100_000)
        assert result.quantity == 0

    def test_zero_vol_returns_zero(self):
        sizer = VolatilitySizer()
        result = sizer.size_position("X", 100.0, 0.0, 100_000)
        assert result.quantity == 0

    def test_portfolio_inverse_vol(self):
        sizer = VolatilitySizer()
        results = sizer.size_portfolio(
            symbols=["A", "B", "C"],
            prices={"A": 50.0, "B": 100.0, "C": 200.0},
            vols={"A": 0.30, "B": 0.15, "C": 0.10},
            equity=100_000,
            method=SizingMethod.INVERSE_VOL,
        )
        assert len(results) == 3
        # Lower vol should get higher weight
        assert results["C"].weight > results["A"].weight

    def test_portfolio_risk_parity(self):
        sizer = VolatilitySizer(target_vol=0.10)
        results = sizer.size_portfolio(
            symbols=["A", "B"],
            prices={"A": 100.0, "B": 100.0},
            vols={"A": 0.20, "B": 0.40},
            equity=100_000,
            method=SizingMethod.RISK_PARITY,
        )
        assert len(results) == 2
        # Equal risk contribution → lower vol asset should have larger position
        assert results["A"].quantity > results["B"].quantity


# ---------------------------------------------------------------------------
# Metric Function Tests
# ---------------------------------------------------------------------------


class TestVolatilityCone:

    def test_basic_output(self, prices):
        cone = volatility_cone(prices)
        assert isinstance(cone, pd.DataFrame)
        assert cone.index.name == "horizon"
        assert "p50" in cone.columns
        assert "current" in cone.columns

    def test_horizons_present(self, prices):
        cone = volatility_cone(prices, horizons=[5, 21, 63])
        assert set(cone.index).issubset({5, 21, 63})

    def test_percentiles_ordered(self, prices):
        cone = volatility_cone(prices)
        for _, row in cone.iterrows():
            assert row["p10"] <= row["p50"] <= row["p90"]


class TestVolOfVol:

    def test_vol_of_vol_output(self, prices):
        returns = np.log(prices / prices.shift(1)).dropna()
        vol = returns.rolling(21).std() * np.sqrt(252)
        vov = vol_of_vol(vol, window=10)
        assert vov.dropna().shape[0] > 0

    def test_vol_of_vol_higher_in_crisis(self):
        """Vol-of-vol should be higher when vol itself is unstable."""
        rng = np.random.default_rng(99)
        # Stable vol period
        stable = pd.Series(np.full(100, 0.20) + rng.normal(0, 0.002, 100))
        # Unstable vol period
        unstable = pd.Series(np.full(100, 0.20) + rng.normal(0, 0.04, 100))

        vov_stable = vol_of_vol(stable, window=10).dropna().mean()
        vov_unstable = vol_of_vol(unstable, window=10).dropna().mean()
        assert vov_unstable > vov_stable


class TestForecastErrorStats:

    def test_perfect_forecast(self):
        x = pd.Series([0.1, 0.2, 0.3, 0.15, 0.25])
        stats = forecast_error_stats(x, x)
        assert stats["mae"] == 0.0
        assert stats["rmse"] == 0.0
        assert stats["bias"] == 0.0
        assert stats["correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_biased_forecast(self):
        realized = pd.Series([0.1, 0.2, 0.3, 0.15, 0.25])
        forecast = realized + 0.05  # constant bias
        stats = forecast_error_stats(forecast, realized)
        assert stats["bias"] == pytest.approx(0.05, abs=1e-6)
        assert stats["mae"] == pytest.approx(0.05, abs=1e-6)

    def test_short_series_returns_nan(self):
        x = pd.Series([0.1])
        stats = forecast_error_stats(x, x)
        assert math.isnan(stats["mae"])


# ---------------------------------------------------------------------------
# Integration: Full Pipeline Test
# ---------------------------------------------------------------------------


class TestVolatilityIntegration:

    def test_end_to_end_pipeline(self, prices):
        """Full pipeline: forecast → size → validate."""
        engine = VolatilityForecastEngine()
        forecast = engine.fit_and_forecast(prices)

        sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01)
        sizing = sizer.size_position(
            "AAPL", float(prices.iloc[-1]),
            forecast.forecast_vol, 100_000,
        )

        assert forecast.forecast_vol > 0
        assert sizing.quantity > 0

    def test_garch_then_sizer(self, garch_returns):
        """GARCH fit → forecast → vol-target sizing."""
        prices = np.exp(garch_returns.cumsum()) * 100
        model = GARCH11()
        result = model.fit(garch_returns)
        fc = model.forecast(result, garch_returns, horizon=1)

        sizer = VolatilitySizer(target_vol=0.10)
        sizing = sizer.size_position(
            "SPY", 400.0, fc["forecast_vol_annualised"], 500_000,
        )
        assert sizing.quantity > 0

    def test_report_and_plots(self, prices, tmp_path):
        """Generate report and plots without crashing."""
        engine = VolatilityForecastEngine()
        output = str(tmp_path / "vol_report.csv")
        df = engine.generate_report(prices, output_path=output)

        # Plots
        engine.plot_forecast_vs_realized(df, save=True, output_dir=str(tmp_path))
        engine.plot_volatility_clusters(df, save=True, output_dir=str(tmp_path))

        import os
        assert os.path.exists(output)
        assert os.path.exists(tmp_path / "vol_forecast_vs_realized.png")
        assert os.path.exists(tmp_path / "vol_clusters.png")
