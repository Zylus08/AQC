"""
examples/run_diagnostics.py
==============================
End-to-end Portfolio Diagnostics & Validation Research.

Runs the full comparative backtest, then performs forensic analysis to answer:
"Why did the portfolio behave the way it did?"

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("diagnostics_research")


def generate_data(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    n1, n2, n3 = n * 40 // 100, n * 15 // 100, n * 25 // 100
    n4 = n - n1 - n2 - n3
    r = np.concatenate([rng.normal(0.0008, 0.006, n1), rng.normal(-0.003, 0.025, n2),
                        rng.normal(0.0, 0.008, n3), rng.normal(0.001, 0.012, n4)])
    prices = 100 * np.exp(np.cumsum(r))
    spread = prices * 0.005
    ohlc = pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": prices + abs(rng.normal(0, 1, n)) * spread,
        "low": prices - abs(rng.normal(0, 1, n)) * spread,
        "close": prices, "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)
    return ohlc, pd.Series(prices, index=idx, name="close")


def simulate_strategy(prices, mode="baseline", initial=100_000, seed=42):
    from aqc.volatility.forecasting_engine import VolatilityForecastEngine
    from aqc.volatility.volatility_metrics import VolatilitySizer
    from aqc.regimes.regime_engine import RegimeEngine, RegimeFilter
    rng = np.random.default_rng(seed)
    n = len(prices)
    vol_engine = VolatilityForecastEngine()
    vol_sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01)
    regime_engine = RegimeEngine(enable_hmm=False)
    regime_filter = RegimeFilter()
    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    zscore = ((prices - ma20) / std20.replace(0, np.nan)).fillna(0)
    equity, position, avg_cost, cash = initial, 0.0, 0.0, initial
    equity_curve, pos_value_list, forecast_vols, realized_vols = [], [], [], []
    for i in range(50, n):
        price = float(prices.iloc[i])
        z = float(zscore.iloc[i])
        sig, direction = 0.0, "HOLD"
        if z < -1.5 and position <= 0: sig, direction = min(abs(z)/3, 1), "LONG"
        elif z > 1.5 and position >= 0: sig, direction = min(abs(z)/3, 1), "SHORT"
        elif position > 0 and z > 0: direction = "EXIT"
        elif position < 0 and z < 0: direction = "EXIT"
        fvol = np.nan
        if direction in ("LONG", "SHORT"):
            qty = 100 * sig
            if mode != "baseline":
                ph = prices.iloc[max(0, i-252):i+1]
                if len(ph) > 50:
                    fc = vol_engine.fit_and_forecast(ph)
                    if fc.forecast_vol > 0:
                        res = vol_sizer.size_position("SIM", price, fc.forecast_vol, equity)
                        qty = max(1, float(res.quantity) * sig)
                        fvol = fc.forecast_vol
                if mode in ("regime_aware", "combined"):
                    ph = prices.iloc[max(0, i-252):i+1]
                    if len(ph) > 60:
                        snapshot = regime_engine.detect(ph)
                        if not regime_filter.should_trade("mean_reversion", snapshot):
                            direction = "HOLD"
            if direction == "LONG" and position <= 0:
                if position < 0:
                    cash += abs(position) * (avg_cost - price) + abs(position) * avg_cost
                    position, avg_cost = 0, 0
                qty = min(qty, cash / price * 0.95)
                if qty >= 1: position, avg_cost = qty, price; cash -= qty * price
            elif direction == "SHORT" and position >= 0:
                if position > 0:
                    cash += position * (price - avg_cost) + position * avg_cost
                    position, avg_cost = 0, 0
                qty = min(qty, cash / price * 0.5)
                if qty >= 1: position, avg_cost = -qty, price; cash += qty * price
        elif direction == "EXIT" and position != 0:
            if position > 0: cash += position * price
            else: cash += abs(position) * (avg_cost - price) + abs(position) * avg_cost
            position, avg_cost = 0, 0
        equity = cash + position * price if position > 0 else cash
        if position < 0: equity = cash - abs(position) * price
        equity_curve.append(equity)
        pos_value_list.append(position * price if position != 0 else 0.0)
        forecast_vols.append(fvol)
        # Realised vol
        if i >= 71:
            recent = np.log(prices.iloc[max(0,i-21):i+1] / prices.iloc[max(0,i-21):i+1].shift(1)).dropna()
            rv = float(recent.std() * np.sqrt(252)) if len(recent) > 2 else np.nan
        else:
            rv = np.nan
        realized_vols.append(rv)
    sub_idx = prices.index[50:]
    eq_s = pd.Series(equity_curve, index=sub_idx, name="equity")
    pv_df = pd.DataFrame({"SIM": pos_value_list}, index=sub_idx)
    fv_s = pd.Series(forecast_vols, index=sub_idx, name="forecast_vol")
    rv_s = pd.Series(realized_vols, index=sub_idx, name="realized_vol")
    return eq_s, pv_df, fv_s, rv_s


def main():
    from aqc.regimes.regime_engine import RegimeEngine
    from aqc.diagnostics.diagnostics_engine import DiagnosticsEngine
    from aqc.diagnostics.diagnostics_report import DiagnosticsReportGenerator
    from aqc.diagnostics.diagnostics_dashboard import PortfolioDiagnosticsDashboard

    print("=" * 70)
    print("AQC PORTFOLIO DIAGNOSTICS & VALIDATION RESEARCH")
    print("=" * 70)
    print()

    # 1. Generate data
    print("[1/6] Generating multi-regime synthetic data...")
    ohlc, prices = generate_data(1000)
    print(f"      {len(prices)} bars | {prices.index[0].date()} to {prices.index[-1].date()}")

    # 2. Detect regimes
    print("[2/6] Detecting regimes...")
    regime_engine = RegimeEngine(enable_hmm=False)
    regime_data = regime_engine.detect_full_series(prices, ohlc_df=ohlc)
    print(f"      Vol regimes: {regime_data['vol_regime'].value_counts().to_dict()}")

    # 3. Run strategies
    print("[3/6] Running comparative backtests...")
    experiments = {}
    for name, mode in [("Baseline", "baseline"), ("Vol-Targeted", "vol_target"),
                       ("Regime-Aware", "regime_aware"), ("Combined", "combined")]:
        print(f"      {name}...")
        eq_s, pv_df, fv_s, rv_s = simulate_strategy(prices, mode)
        experiments[name] = {"equity": eq_s, "pos_values": pv_df, "forecast_vol": fv_s, "realised_vol": rv_s}
    print()

    # 4. Run diagnostics on each experiment
    print("[4/6] Running portfolio diagnostics...")
    Path("reports").mkdir(exist_ok=True)
    for name, exp in experiments.items():
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTICS: {name}")
        print(f"{'='*60}")
        eq = exp["equity"]
        returns = eq.pct_change().dropna()
        out_dir = f"reports/diagnostics_{name.lower().replace('-', '_').replace(' ', '_')}"

        # Build vol data for forecast validation
        vol_data = pd.DataFrame({
            "ensemble_vol": exp["forecast_vol"],
            "ewma_vol": exp["forecast_vol"] * 0.95,
            "garch_vol": exp["forecast_vol"] * 1.05,
            "hist_vol": exp["realised_vol"],
            "realized_1d": exp["realised_vol"],
        }, index=eq.index)

        engine = DiagnosticsEngine(
            equity_series=eq,
            position_values=exp["pos_values"],
            returns=returns,
            regime_data=regime_data.reindex(eq.index).ffill(),
            vol_data=vol_data,
            forecast_vol_series=exp["forecast_vol"],
            realised_vol_series=exp["realised_vol"],
            baseline_returns=experiments["Baseline"]["equity"].pct_change().dropna() if name != "Baseline" else returns,
            target_vol=0.10,
            max_leverage=3.0,
        )
        results = engine.run_all(output_dir=out_dir)

        # Print report
        reporter = DiagnosticsReportGenerator(results)
        reporter.print_report()
        reporter.save_summary(f"{out_dir}/diagnostics_summary.csv")

    # 5. Generate dashboards
    print("\n[5/6] Generating HTML dashboards...")
    for name in experiments:
        out_dir = f"reports/diagnostics_{name.lower().replace('-', '_').replace(' ', '_')}"
        eq = experiments[name]["equity"]
        returns = eq.pct_change().dropna()
        vol_data = pd.DataFrame({
            "ensemble_vol": experiments[name]["forecast_vol"],
            "realized_1d": experiments[name]["realised_vol"],
        }, index=eq.index)
        engine = DiagnosticsEngine(
            eq, experiments[name]["pos_values"], returns,
            regime_data=regime_data.reindex(eq.index).ffill(),
            vol_data=vol_data,
            forecast_vol_series=experiments[name]["forecast_vol"],
            realised_vol_series=experiments[name]["realised_vol"],
        )
        results = engine.run_all(output_dir=out_dir)
        dash = PortfolioDiagnosticsDashboard(results, out_dir,
               f"dashboard/diagnostics_{name.lower().replace('-', '_').replace(' ', '_')}.html")
        dash.generate()
        print(f"      Saved: dashboard/diagnostics_{name.lower().replace('-', '_').replace(' ', '_')}.html")

    # 6. Cross-experiment comparison
    print("\n[6/6] Cross-Experiment Validation Summary")
    print("-" * 70)
    print(f"  {'Experiment':<20s} {'Sharpe':>8s} {'Max Lev':>8s} {'Risk Util':>10s} {'Val Score':>10s}")
    print("-" * 70)
    for name, exp in experiments.items():
        eq = exp["equity"]
        returns = eq.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        # Quick leverage
        pv = exp["pos_values"]
        lev = pv.abs().sum(axis=1) / eq.reindex(pv.index).ffill()
        max_lev = float(lev.max())
        # Quick risk util (placeholder)
        print(f"  {name:<20s} {sharpe:>8.2f} {max_lev:>8.2f} {'N/A':>10s} {'--':>10s}")

    print()
    print("=" * 70)
    print("Diagnostics complete. Reports: reports/diagnostics_*/")
    print("Dashboards: dashboard/diagnostics_*.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
