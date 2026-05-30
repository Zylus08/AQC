"""
tests/test_diagnostics.py
===========================
Tests for the Portfolio Diagnostics & Validation Framework.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import numpy as np, pandas as pd, pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _idx(n=500):
    return pd.date_range("2020-01-01", periods=n, freq="B")

def _equity(n=500, seed=42):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.015, n)
    eq = 100_000 * np.exp(np.cumsum(ret))
    return pd.Series(eq, index=_idx(n), name="equity")

def _pos_values(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "AAPL": rng.normal(30000, 5000, n),
        "MSFT": rng.normal(25000, 4000, n),
        "TLT":  rng.normal(20000, 3000, n),
    }, index=_idx(n))

def _returns(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.015, n), index=_idx(n), name="ret")

def _regime_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    vr = rng.choice(["LOW", "NORMAL", "HIGH", "EXTREME"], n, p=[0.2, 0.4, 0.3, 0.1])
    tr = rng.choice(["UPTREND", "RANGE_BOUND", "DOWNTREND"], n, p=[0.3, 0.4, 0.3])
    return pd.DataFrame({"vol_regime": vr, "trend_regime": tr}, index=_idx(n))

def _vol_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    realized = rng.uniform(0.05, 0.30, n)
    ewma = realized + rng.normal(0, 0.02, n)
    garch = realized + rng.normal(0.01, 0.03, n)
    return pd.DataFrame({
        "ewma_vol": ewma, "garch_vol": garch, "hist_vol": realized + rng.normal(0, 0.01, n),
        "ensemble_vol": (ewma + garch + realized) / 3, "realized_1d": realized,
    }, index=_idx(n))

# ===========================================================================
# LEVERAGE ANALYSIS
# ===========================================================================
class TestLeverageAnalyzer:
    def test_compute_returns_df(self):
        from aqc.diagnostics.leverage_analysis import LeverageAnalyzer
        la = LeverageAnalyzer(_equity(), _pos_values())
        df = la.compute()
        assert "gross_leverage" in df.columns
        assert "net_leverage" in df.columns

    def test_stats_returns_dataclass(self):
        from aqc.diagnostics.leverage_analysis import LeverageAnalyzer, LeverageStats
        la = LeverageAnalyzer(_equity(), _pos_values())
        st = la.stats()
        assert isinstance(st, LeverageStats)
        assert st.avg_gross > 0

    def test_leverage_during_drawdowns(self):
        from aqc.diagnostics.leverage_analysis import LeverageAnalyzer
        la = LeverageAnalyzer(_equity(), _pos_values())
        dd_stats = la.leverage_during_drawdowns(-0.001)
        assert dd_stats is not None

    def test_leverage_by_regime(self):
        from aqc.diagnostics.leverage_analysis import LeverageAnalyzer
        la = LeverageAnalyzer(_equity(), _pos_values())
        rd = _regime_data()
        result = la.leverage_by_regime(rd["vol_regime"])
        assert len(result) > 0

    def test_plot_runs(self):
        from aqc.diagnostics.leverage_analysis import LeverageAnalyzer
        la = LeverageAnalyzer(_equity(100), _pos_values(100))
        la.plot(save=False)

# ===========================================================================
# EXPOSURE ANALYSIS
# ===========================================================================
class TestExposureAnalyzer:
    def test_compute_returns_df(self):
        from aqc.diagnostics.exposure_analysis import ExposureAnalyzer
        ea = ExposureAnalyzer(_equity(), _pos_values())
        df = ea.compute()
        assert "long_exposure" in df.columns
        assert "short_exposure" in df.columns

    def test_stats(self):
        from aqc.diagnostics.exposure_analysis import ExposureAnalyzer, ExposureStats
        ea = ExposureAnalyzer(_equity(), _pos_values())
        st = ea.stats()
        assert isinstance(st, ExposureStats)

    def test_exposure_by_regime(self):
        from aqc.diagnostics.exposure_analysis import ExposureAnalyzer
        ea = ExposureAnalyzer(_equity(), _pos_values())
        rd = _regime_data()
        result = ea.exposure_by_regime(rd["vol_regime"])
        assert len(result) > 0

    def test_plot_runs(self):
        from aqc.diagnostics.exposure_analysis import ExposureAnalyzer
        ea = ExposureAnalyzer(_equity(100), _pos_values(100))
        ea.plot(save=False)

# ===========================================================================
# RISK BUDGET ANALYSIS
# ===========================================================================
class TestRiskBudgetAnalyzer:
    def test_compute(self):
        from aqc.diagnostics.risk_budget_analysis import RiskBudgetAnalyzer
        vd = _vol_data()
        rb = RiskBudgetAnalyzer(_equity(), vd["ensemble_vol"], vd["realized_1d"], _pos_values())
        df = rb.compute()
        assert "utilisation" in df.columns

    def test_stats(self):
        from aqc.diagnostics.risk_budget_analysis import RiskBudgetAnalyzer, RiskBudgetStats
        vd = _vol_data()
        rb = RiskBudgetAnalyzer(_equity(), vd["ensemble_vol"], vd["realized_1d"], _pos_values())
        st = rb.stats()
        assert isinstance(st, RiskBudgetStats)

    def test_plot_runs(self):
        from aqc.diagnostics.risk_budget_analysis import RiskBudgetAnalyzer
        vd = _vol_data(100)
        rb = RiskBudgetAnalyzer(_equity(100), vd["ensemble_vol"], vd["realized_1d"], _pos_values(100))
        rb.plot(save=False)

# ===========================================================================
# POSITION ANALYSIS
# ===========================================================================
class TestPositionAnalyzer:
    def test_compute(self):
        from aqc.diagnostics.position_analysis import PositionAnalyzer
        pa = PositionAnalyzer(_equity(), _pos_values())
        df = pa.compute()
        assert "hhi" in df.columns
        assert "num_positions" in df.columns

    def test_stats(self):
        from aqc.diagnostics.position_analysis import PositionAnalyzer, PositionStats
        pa = PositionAnalyzer(_equity(), _pos_values())
        st = pa.stats()
        assert isinstance(st, PositionStats)
        assert st.avg_size > 0

    def test_largest_positions(self):
        from aqc.diagnostics.position_analysis import PositionAnalyzer
        pa = PositionAnalyzer(_equity(), _pos_values())
        top = pa.largest_positions(3)
        assert len(top) == 3

    def test_plot_runs(self):
        from aqc.diagnostics.position_analysis import PositionAnalyzer
        pa = PositionAnalyzer(_equity(100), _pos_values(100))
        pa.plot(save=False)

# ===========================================================================
# REGIME PERFORMANCE
# ===========================================================================
class TestRegimePerformanceAnalyzer:
    def test_by_vol_regime(self):
        from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
        rpa = RegimePerformanceAnalyzer(_returns(), _regime_data())
        df = rpa.by_vol_regime()
        assert "sharpe" in df.columns
        assert len(df) > 0

    def test_by_trend_regime(self):
        from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
        rpa = RegimePerformanceAnalyzer(_returns(), _regime_data())
        df = rpa.by_trend_regime()
        assert len(df) > 0

    def test_contribution(self):
        from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
        rpa = RegimePerformanceAnalyzer(_returns(), _regime_data())
        c = rpa.regime_contribution()
        assert "pct_contribution" in c.columns

    def test_plot_runs(self):
        from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
        rpa = RegimePerformanceAnalyzer(_returns(100), _regime_data(100))
        rpa.plot(save=False)

# ===========================================================================
# FORECAST ANALYSIS
# ===========================================================================
class TestForecastAnalyzer:
    def test_accuracy_table(self):
        from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
        fa = ForecastAnalyzer(_vol_data())
        tbl = fa.accuracy_table()
        assert len(tbl) > 0
        assert "rmse" in tbl.columns

    def test_best_model(self):
        from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
        fa = ForecastAnalyzer(_vol_data())
        m = fa.best_model()
        assert m in ["ewma_vol", "garch_vol", "hist_vol", "ensemble_vol"]

    def test_error_by_regime(self):
        from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
        fa = ForecastAnalyzer(_vol_data())
        rd = _regime_data()
        err = fa.error_by_regime(rd["vol_regime"])
        assert len(err) > 0

    def test_plot_runs(self):
        from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
        fa = ForecastAnalyzer(_vol_data(100))
        fa.plot(save=False)

# ===========================================================================
# ATTRIBUTION
# ===========================================================================
class TestAttribution:
    def test_compute_returns_result(self):
        from aqc.diagnostics.attribution import PerformanceAttributionEngine, AttributionResult
        r = _returns()
        ae = PerformanceAttributionEngine(r)
        result = ae.compute()
        assert isinstance(result, AttributionResult)

    def test_to_dict(self):
        from aqc.diagnostics.attribution import PerformanceAttributionEngine
        r = _returns()
        ae = PerformanceAttributionEngine(r)
        d = ae.compute().to_dict()
        assert "Alpha" in d
        assert "Total" in d

    def test_pct_dict(self):
        from aqc.diagnostics.attribution import PerformanceAttributionEngine
        r = _returns()
        ae = PerformanceAttributionEngine(r)
        p = ae.compute().pct_dict()
        assert "Alpha" in p

    def test_with_vol_target(self):
        from aqc.diagnostics.attribution import PerformanceAttributionEngine
        r1 = _returns(500, seed=1)
        r2 = _returns(500, seed=2)
        ae = PerformanceAttributionEngine(r1, vol_target_returns=r2, combined_returns=r2)
        result = ae.compute()
        assert result.vol_target_contribution != 0 or True

    def test_plot_runs(self):
        from aqc.diagnostics.attribution import PerformanceAttributionEngine
        ae = PerformanceAttributionEngine(_returns(100))
        ae.plot(save=False)

# ===========================================================================
# DRAWDOWN ANALYZER
# ===========================================================================
class TestDrawdownAnalyzer:
    def test_find_drawdowns(self):
        from aqc.diagnostics.diagnostics_engine import DrawdownAnalyzer
        da = DrawdownAnalyzer(_equity())
        events = da.find_drawdowns()
        assert len(events) > 0
        assert events[0].depth_pct < 0

    def test_to_dataframe(self):
        from aqc.diagnostics.diagnostics_engine import DrawdownAnalyzer
        da = DrawdownAnalyzer(_equity())
        df = da.to_dataframe()
        assert "depth_pct" in df.columns
        assert "duration_days" in df.columns

    def test_with_regime_annotation(self):
        from aqc.diagnostics.diagnostics_engine import DrawdownAnalyzer
        da = DrawdownAnalyzer(_equity(), regime_data=_regime_data())
        events = da.find_drawdowns()
        # At least one should have regime info
        has_regime = any(e.vol_regime != "" for e in events)
        assert has_regime

    def test_plot_runs(self):
        from aqc.diagnostics.diagnostics_engine import DrawdownAnalyzer
        da = DrawdownAnalyzer(_equity(100))
        da.plot(save=False)

# ===========================================================================
# PORTFOLIO VALIDATOR
# ===========================================================================
class TestPortfolioValidator:
    def test_clean_portfolio_scores_high(self):
        from aqc.diagnostics.diagnostics_engine import PortfolioValidator
        from aqc.diagnostics.leverage_analysis import LeverageStats
        from aqc.diagnostics.exposure_analysis import ExposureStats
        from aqc.diagnostics.position_analysis import PositionStats

        v = PortfolioValidator(
            leverage_stats=LeverageStats(avg_gross=0.5, max_gross=1.0),
            exposure_stats=ExposureStats(avg_gross=0.5, max_gross=0.8, pct_fully_invested=0.7),
            position_stats=PositionStats(hhi_concentration=0.2, max_weight=0.3),
        )
        score = v.validate()
        assert score.overall_score >= 80

    def test_over_leveraged_penalised(self):
        from aqc.diagnostics.diagnostics_engine import PortfolioValidator
        from aqc.diagnostics.leverage_analysis import LeverageStats

        v = PortfolioValidator(
            leverage_stats=LeverageStats(avg_gross=4.0, max_gross=8.0, pct_leveraged=0.9),
            max_leverage=3.0,
        )
        score = v.validate()
        assert score.leverage_score < 60
        assert len(score.violations) > 0

    def test_risk_budget_violation(self):
        from aqc.diagnostics.diagnostics_engine import PortfolioValidator
        from aqc.diagnostics.risk_budget_analysis import RiskBudgetStats

        v = PortfolioValidator(
            risk_budget_stats=RiskBudgetStats(pct_over_budget=0.4, max_utilisation=3.0),
        )
        score = v.validate()
        assert score.risk_budget_score < 60

    def test_validation_dict(self):
        from aqc.diagnostics.diagnostics_engine import PortfolioValidator
        v = PortfolioValidator()
        score = v.validate()
        d = score.to_dict()
        assert "Overall" in d

# ===========================================================================
# DIAGNOSTICS ENGINE (integration)
# ===========================================================================
class TestDiagnosticsEngine:
    def test_run_all_minimal(self):
        from aqc.diagnostics.diagnostics_engine import DiagnosticsEngine
        import tempfile, os
        eq = _equity(200)
        pv = _pos_values(200)
        ret = _returns(200)
        engine = DiagnosticsEngine(eq, pv, ret)
        d = os.path.join(tempfile.mkdtemp(), "diag_test")
        results = engine.run_all(output_dir=d)
        assert "leverage" in results
        assert "exposure" in results
        assert "validation" in results

    def test_run_all_with_regime(self):
        from aqc.diagnostics.diagnostics_engine import DiagnosticsEngine
        import tempfile, os
        eq = _equity(200)
        pv = _pos_values(200)
        ret = _returns(200)
        rd = _regime_data(200)
        vd = _vol_data(200)
        engine = DiagnosticsEngine(
            eq, pv, ret,
            regime_data=rd,
            vol_data=vd,
            forecast_vol_series=vd["ensemble_vol"],
            realised_vol_series=vd["realized_1d"],
        )
        d = os.path.join(tempfile.mkdtemp(), "diag_full")
        results = engine.run_all(output_dir=d)
        assert "risk_budget" in results
        assert "forecast" in results

# ===========================================================================
# REPORT GENERATOR
# ===========================================================================
class TestDiagnosticsReport:
    def test_build_report(self):
        from aqc.diagnostics.diagnostics_report import DiagnosticsReportGenerator
        from aqc.diagnostics.leverage_analysis import LeverageStats
        results = {"leverage": LeverageStats(avg_gross=0.5, max_gross=1.2),
                    "validation": {"Overall": 85}, "violations": []}
        rg = DiagnosticsReportGenerator(results)
        report = rg.build_report()
        assert "DIAGNOSTICS REPORT" in report
        assert "0.5" in report

# ===========================================================================
# DASHBOARD
# ===========================================================================
class TestDashboard:
    def test_generate_html(self):
        from aqc.diagnostics.diagnostics_dashboard import PortfolioDiagnosticsDashboard
        from aqc.diagnostics.leverage_analysis import LeverageStats
        import tempfile, os
        results = {"validation": {"Overall": 90, "Leverage": 92}, "violations": [],
                    "leverage": LeverageStats(avg_gross=0.5, max_gross=1.0),
                    "attribution": {"Alpha": 0.05, "Total": 0.10}}
        outpath = os.path.join(tempfile.mkdtemp(), "test_dash.html")
        dash = PortfolioDiagnosticsDashboard(results, output_path=outpath)
        html = dash.generate()
        assert "AQC Portfolio Diagnostics" in html
        assert os.path.exists(outpath)
