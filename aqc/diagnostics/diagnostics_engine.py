"""
aqc/diagnostics/diagnostics_engine.py
=======================================
Composite diagnostics engine + PortfolioValidator (automated health checks).

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging, math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np, pandas as pd

from aqc.diagnostics.leverage_analysis import LeverageAnalyzer, LeverageStats
from aqc.diagnostics.exposure_analysis import ExposureAnalyzer, ExposureStats
from aqc.diagnostics.risk_budget_analysis import RiskBudgetAnalyzer, RiskBudgetStats
from aqc.diagnostics.position_analysis import PositionAnalyzer, PositionStats
from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
from aqc.diagnostics.attribution import PerformanceAttributionEngine, AttributionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DrawdownAnalyzer (Part 9)
# ---------------------------------------------------------------------------

@dataclass
class DrawdownEvent:
    """Single drawdown episode."""
    start: pd.Timestamp
    trough: pd.Timestamp
    end: Optional[pd.Timestamp]
    depth_pct: float
    duration_days: int
    recovery_days: Optional[int]
    avg_leverage: float = 0.0
    avg_exposure: float = 0.0
    vol_regime: str = ""
    trend_regime: str = ""
    avg_forecast_error: float = 0.0


class DrawdownAnalyzer:
    """Forensic analysis of drawdown episodes.

    Parameters
    ----------
    equity_series : pd.Series
    leverage_series : pd.Series (optional)
    exposure_series : pd.Series (optional)
    regime_data : pd.DataFrame (optional)
    forecast_error : pd.Series (optional)
    top_n : int  number of worst drawdowns to analyse
    """
    def __init__(self, equity_series: pd.Series,
                 leverage_series: Optional[pd.Series] = None,
                 exposure_series: Optional[pd.Series] = None,
                 regime_data: Optional[pd.DataFrame] = None,
                 forecast_error: Optional[pd.Series] = None,
                 top_n: int = 5) -> None:
        self.equity = equity_series
        self.leverage = leverage_series
        self.exposure = exposure_series
        self.regime_data = regime_data
        self.forecast_error = forecast_error
        self.top_n = top_n

    def find_drawdowns(self) -> list[DrawdownEvent]:
        eq = self.equity
        peak = eq.cummax()
        dd = (eq - peak) / peak
        in_dd = dd < -0.001
        events = []
        start = None
        for i in range(len(dd)):
            if in_dd.iloc[i] and start is None:
                start = dd.index[i]
            elif not in_dd.iloc[i] and start is not None:
                mask = (dd.index >= start) & (dd.index <= dd.index[i])
                segment = dd.loc[mask]
                trough_idx = segment.idxmin()
                depth = float(segment.min()) * 100
                dur = (trough_idx - start).days
                rec = (dd.index[i] - trough_idx).days
                evt = DrawdownEvent(
                    start=start, trough=trough_idx, end=dd.index[i],
                    depth_pct=round(depth, 4), duration_days=dur,
                    recovery_days=rec,
                )
                self._annotate(evt, start, dd.index[i])
                events.append(evt)
                start = None
        # Handle open drawdown
        if start is not None:
            mask = dd.index >= start
            segment = dd.loc[mask]
            trough_idx = segment.idxmin()
            depth = float(segment.min()) * 100
            dur = (trough_idx - start).days
            evt = DrawdownEvent(
                start=start, trough=trough_idx, end=None,
                depth_pct=round(depth, 4), duration_days=dur,
                recovery_days=None,
            )
            self._annotate(evt, start, dd.index[-1])
            events.append(evt)
        events.sort(key=lambda e: e.depth_pct)
        return events[:self.top_n]

    def _annotate(self, evt: DrawdownEvent, start, end) -> None:
        if self.leverage is not None:
            lev = self.leverage.loc[start:end]
            if len(lev) > 0:
                evt.avg_leverage = round(float(lev.mean()), 4)
        if self.exposure is not None:
            exp = self.exposure.loc[start:end]
            if len(exp) > 0:
                evt.avg_exposure = round(float(exp.mean()), 4)
        if self.regime_data is not None:
            rd = self.regime_data.loc[start:end]
            if len(rd) > 0 and "vol_regime" in rd.columns:
                evt.vol_regime = str(rd["vol_regime"].mode().iloc[0]) if len(rd["vol_regime"].mode()) > 0 else ""
            if len(rd) > 0 and "trend_regime" in rd.columns:
                evt.trend_regime = str(rd["trend_regime"].mode().iloc[0]) if len(rd["trend_regime"].mode()) > 0 else ""
        if self.forecast_error is not None:
            fe = self.forecast_error.loc[start:end]
            if len(fe) > 0:
                evt.avg_forecast_error = round(float(fe.mean()), 6)

    def to_dataframe(self) -> pd.DataFrame:
        events = self.find_drawdowns()
        rows = []
        for e in events:
            rows.append({
                "start": e.start, "trough": e.trough, "end": e.end,
                "depth_pct": e.depth_pct, "duration_days": e.duration_days,
                "recovery_days": e.recovery_days, "avg_leverage": e.avg_leverage,
                "avg_exposure": e.avg_exposure, "vol_regime": e.vol_regime,
                "trend_regime": e.trend_regime, "avg_forecast_error": e.avg_forecast_error,
            })
        return pd.DataFrame(rows)

    def save_report(self, path: str = "reports/drawdown_forensics_report.csv") -> None:
        from pathlib import Path
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def plot(self, save: bool = True, output_dir: str = "reports") -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        eq = self.equity
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        events = self.find_drawdowns()

        fig, ax = plt.subplots(figsize=(16, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.fill_between(dd.index, dd, 0, color="#EF5350", alpha=0.4)
        ax.plot(dd.index, dd, color="#EF5350", lw=1.0)
        for e in events[:3]:
            ax.annotate(f"{e.depth_pct:.1f}%",
                       xy=(e.trough, e.depth_pct),
                       xytext=(e.trough, e.depth_pct - 2),
                       color="white", fontsize=9, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
        ax.set_title("Drawdown Forensics", color="white", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", color="white")
        for s in ax.spines.values(): s.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "drawdown_forensics.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


# ---------------------------------------------------------------------------
# PortfolioValidator (Part 11)
# ---------------------------------------------------------------------------

@dataclass
class ValidationScore:
    leverage_score: int = 100
    risk_budget_score: int = 100
    exposure_score: int = 100
    concentration_score: int = 100
    forecast_score: int = 100
    regime_score: int = 100
    overall_score: int = 100
    violations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "Leverage": self.leverage_score,
            "Risk Budget": self.risk_budget_score,
            "Exposure": self.exposure_score,
            "Concentration": self.concentration_score,
            "Forecasting": self.forecast_score,
            "Regime": self.regime_score,
            "Overall": self.overall_score,
        }


class PortfolioValidator:
    """Automated portfolio health checks with scoring (0-100).

    Parameters
    ----------
    leverage_stats : LeverageStats (optional)
    risk_budget_stats : RiskBudgetStats (optional)
    exposure_stats : ExposureStats (optional)
    position_stats : PositionStats (optional)
    forecast_accuracy : list (optional)  from ForecastAnalyzer
    max_leverage : float
    target_vol : float
    """
    def __init__(
        self,
        leverage_stats: Optional[LeverageStats] = None,
        risk_budget_stats: Optional[RiskBudgetStats] = None,
        exposure_stats: Optional[ExposureStats] = None,
        position_stats: Optional[PositionStats] = None,
        forecast_accuracy: Optional[list] = None,
        max_leverage: float = 3.0,
        target_vol: float = 0.10,
    ) -> None:
        self.leverage_stats = leverage_stats
        self.risk_budget_stats = risk_budget_stats
        self.exposure_stats = exposure_stats
        self.position_stats = position_stats
        self.forecast_accuracy = forecast_accuracy or []
        self.max_leverage = max_leverage
        self.target_vol = target_vol

    def validate(self) -> ValidationScore:
        score = ValidationScore()

        # Leverage
        if self.leverage_stats:
            ls = self.leverage_stats
            if ls.max_gross > self.max_leverage:
                penalty = min(40, int((ls.max_gross - self.max_leverage) / self.max_leverage * 100))
                score.leverage_score -= penalty
                score.violations.append(f"Max leverage {ls.max_gross:.2f} > limit {self.max_leverage}")
            if ls.pct_leveraged > 0.5:
                score.leverage_score -= 15
                score.violations.append(f"Leveraged {ls.pct_leveraged*100:.0f}% of time")
            score.leverage_score = max(0, score.leverage_score)

        # Risk Budget
        if self.risk_budget_stats:
            rb = self.risk_budget_stats
            if rb.pct_over_budget > 0.1:
                penalty = min(30, int(rb.pct_over_budget * 100))
                score.risk_budget_score -= penalty
                score.violations.append(f"Over-budget {rb.pct_over_budget*100:.0f}% of time")
            if rb.max_utilisation > 2.0:
                score.risk_budget_score -= 20
                score.violations.append(f"Max risk utilisation {rb.max_utilisation:.2f}x")
            score.risk_budget_score = max(0, score.risk_budget_score)

        # Exposure
        if self.exposure_stats:
            es = self.exposure_stats
            if es.max_gross > 2.0:
                score.exposure_score -= 20
                score.violations.append(f"Max gross exposure {es.max_gross:.2f}")
            if es.pct_fully_invested < 0.3:
                score.exposure_score -= 10
                score.violations.append(f"Only fully invested {es.pct_fully_invested*100:.0f}% of time")
            score.exposure_score = max(0, score.exposure_score)

        # Concentration
        if self.position_stats:
            ps = self.position_stats
            if ps.hhi_concentration > 0.5:
                penalty = min(30, int((ps.hhi_concentration - 0.5) * 100))
                score.concentration_score -= penalty
                score.violations.append(f"HHI concentration {ps.hhi_concentration:.4f}")
            if ps.max_weight > 0.5:
                score.concentration_score -= 15
                score.violations.append(f"Max position weight {ps.max_weight:.2%}")
            score.concentration_score = max(0, score.concentration_score)

        # Forecast
        if self.forecast_accuracy:
            best = min(self.forecast_accuracy, key=lambda a: a.rmse)
            if best.rmse > 0.05:
                penalty = min(30, int(best.rmse * 200))
                score.forecast_score -= penalty
                score.violations.append(f"Best RMSE {best.rmse:.4f}")
            if best.correlation < 0.5:
                score.forecast_score -= 15
                score.violations.append(f"Best corr {best.correlation:.2f}")
            score.forecast_score = max(0, score.forecast_score)

        # Regime
        score.regime_score = 90  # default good unless specific issues
        if self.risk_budget_stats and self.risk_budget_stats.pct_over_budget > 0.2:
            score.regime_score -= 10

        # Overall
        scores = [score.leverage_score, score.risk_budget_score, score.exposure_score,
                  score.concentration_score, score.forecast_score, score.regime_score]
        score.overall_score = int(np.mean(scores))
        return score


# ---------------------------------------------------------------------------
# DiagnosticsEngine (Part 10 — composite)
# ---------------------------------------------------------------------------


class DiagnosticsEngine:
    """Composite forensic analysis orchestrator.

    Parameters
    ----------
    equity_series : pd.Series
    position_values : pd.DataFrame  cols=symbols, vals=signed market value
    returns : pd.Series             daily portfolio returns
    trade_log : list[dict]
    regime_data : pd.DataFrame (opt)
    vol_data : pd.DataFrame (opt)   forecast vol columns + realized
    forecast_vol_series : pd.Series (opt)
    realised_vol_series : pd.Series (opt)
    baseline_returns : pd.Series (opt)   for attribution
    vol_target_returns : pd.Series (opt)
    regime_returns : pd.Series (opt)
    initial_capital : float
    target_vol : float
    max_leverage : float
    """
    def __init__(
        self,
        equity_series: pd.Series,
        position_values: pd.DataFrame,
        returns: pd.Series,
        trade_log: Optional[list[dict]] = None,
        regime_data: Optional[pd.DataFrame] = None,
        vol_data: Optional[pd.DataFrame] = None,
        forecast_vol_series: Optional[pd.Series] = None,
        realised_vol_series: Optional[pd.Series] = None,
        baseline_returns: Optional[pd.Series] = None,
        vol_target_returns: Optional[pd.Series] = None,
        regime_returns: Optional[pd.Series] = None,
        initial_capital: float = 100_000.0,
        target_vol: float = 0.10,
        max_leverage: float = 3.0,
    ) -> None:
        self.equity = equity_series
        self.pos_values = position_values
        self.returns = returns
        self.trade_log = trade_log or []
        self.regime_data = regime_data
        self.vol_data = vol_data
        self.forecast_vol = forecast_vol_series
        self.realised_vol = realised_vol_series
        self.baseline_returns = baseline_returns
        self.vol_target_returns = vol_target_returns
        self.regime_returns = regime_returns
        self.initial_capital = initial_capital
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def run_all(self, output_dir: str = "reports") -> dict:
        """Execute all diagnostics and return summary dict."""
        results = {}

        # Leverage
        lev = LeverageAnalyzer(self.equity, self.pos_values, self.initial_capital, self.max_leverage)
        lev_df = lev.compute()
        lev.save_report(f"{output_dir}/leverage_report.csv")
        lev.plot(output_dir=output_dir)
        results["leverage"] = lev.stats()

        # Exposure
        exp = ExposureAnalyzer(self.equity, self.pos_values)
        exp.compute()
        exp.save_report(f"{output_dir}/exposure_report.csv")
        exp.plot(output_dir=output_dir)
        results["exposure"] = exp.stats()

        # Risk Budget
        if self.forecast_vol is not None and self.realised_vol is not None:
            rb = RiskBudgetAnalyzer(self.equity, self.forecast_vol, self.realised_vol,
                                   self.pos_values, self.target_vol)
            rb.compute()
            rb.save_report(f"{output_dir}/risk_budget_report.csv")
            rb.plot(output_dir=output_dir)
            results["risk_budget"] = rb.stats()

        # Position
        pos = PositionAnalyzer(self.equity, self.pos_values, self.trade_log)
        pos.compute()
        pos.save_report(f"{output_dir}/position_analysis.csv")
        pos.plot(output_dir=output_dir)
        results["position"] = pos.stats()

        # Regime Performance
        if self.regime_data is not None:
            rpa = RegimePerformanceAnalyzer(self.returns, self.regime_data)
            rpa.save_report(f"{output_dir}/regime_performance_report.csv")
            rpa.plot(output_dir=output_dir)
            results["regime_vol"] = rpa.by_vol_regime().to_dict()
            results["regime_trend"] = rpa.by_trend_regime().to_dict()

        # Forecast Validation
        if self.vol_data is not None:
            fa = ForecastAnalyzer(self.vol_data)
            fa.save_report(f"{output_dir}/forecast_validation_report.csv")
            fa.plot(output_dir=output_dir)
            results["forecast"] = fa.accuracy_table().to_dict() if not fa.accuracy_table().empty else {}

        # Attribution
        attr = PerformanceAttributionEngine(
            baseline_returns=self.baseline_returns if self.baseline_returns is not None else self.returns,
            vol_target_returns=self.vol_target_returns,
            regime_returns=self.regime_returns,
            combined_returns=self.returns,
            leverage_series=lev_df["gross_leverage"] if lev_df is not None else None,
        )
        attr.save_report(f"{output_dir}/attribution_report.csv")
        attr.plot(output_dir=output_dir)
        results["attribution"] = attr.compute().to_dict()

        # Drawdown Forensics
        dd = DrawdownAnalyzer(
            self.equity,
            leverage_series=lev_df["gross_leverage"] if lev_df is not None else None,
            exposure_series=lev_df["gross_exposure"] if lev_df is not None else None,
            regime_data=self.regime_data,
        )
        dd.save_report(f"{output_dir}/drawdown_forensics_report.csv")
        dd.plot(output_dir=output_dir)
        results["drawdowns"] = dd.to_dataframe().to_dict()

        # Validation
        forecast_acc = ForecastAnalyzer(self.vol_data).compute_accuracy() if self.vol_data is not None else []
        validator = PortfolioValidator(
            leverage_stats=results.get("leverage"),
            risk_budget_stats=results.get("risk_budget"),
            exposure_stats=results.get("exposure"),
            position_stats=results.get("position"),
            forecast_accuracy=forecast_acc,
            max_leverage=self.max_leverage,
            target_vol=self.target_vol,
        )
        validation = validator.validate()
        results["validation"] = validation.to_dict()
        results["violations"] = validation.violations

        # Save validation
        from pathlib import Path
        val_df = pd.DataFrame([validation.to_dict()])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        val_df.to_csv(f"{output_dir}/validation_report.csv", index=False)

        logger.info("Diagnostics complete — %d reports generated in %s", 9, output_dir)
        return results
