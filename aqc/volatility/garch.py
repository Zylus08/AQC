"""
aqc/volatility/garch.py
========================
GARCH(1,1) Volatility Model.

Implements the standard Generalised Autoregressive Conditional
Heteroskedasticity model of Bollerslev (1986):

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

where:
- omega > 0 is the constant (drives long-run variance)
- alpha >= 0 is the ARCH coefficient (shock impact)
- beta >= 0 is the GARCH coefficient (persistence)
- alpha + beta < 1 ensures stationarity

The long-run (unconditional) variance is:

    sigma^2_LR = omega / (1 - alpha - beta)

Parameter estimation uses maximum-likelihood with Gaussian log-likelihood,
optimised via scipy's L-BFGS-B.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Container for fitted GARCH(1,1) parameters.

    Attributes
    ----------
    omega:
        Constant term.
    alpha:
        ARCH coefficient (reaction to shocks).
    beta:
        GARCH coefficient (persistence).
    log_likelihood:
        Maximised log-likelihood value.
    long_run_variance:
        Unconditional variance: omega / (1 - alpha - beta).
    long_run_volatility:
        Annualised long-run volatility.
    persistence:
        alpha + beta (< 1 for stationarity).
    half_life:
        Half-life of volatility shocks in periods.
    conditional_variance:
        In-sample fitted variance series.
    converged:
        Whether the optimiser converged.
    """

    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    log_likelihood: float = 0.0
    long_run_variance: float = 0.0
    long_run_volatility: float = 0.0
    persistence: float = 0.0
    half_life: float = 0.0
    conditional_variance: Optional[pd.Series] = None
    converged: bool = False

    def __repr__(self) -> str:
        return (
            f"GARCHResult(omega={self.omega:.6f}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f}, "
            f"LR_vol={self.long_run_volatility:.4f})"
        )


class GARCH11:
    """GARCH(1,1) volatility model with MLE fitting.

    Parameters
    ----------
    ann_factor:
        Annualisation factor (252 for daily).

    Examples
    --------
    >>> model = GARCH11()
    >>> result = model.fit(log_returns)
    >>> forecast = model.forecast(result, horizon=5)
    """

    def __init__(self, ann_factor: int = 252) -> None:
        self.ann_factor = ann_factor

    def fit(
        self,
        returns: pd.Series,
        initial_params: Optional[tuple[float, float, float]] = None,
    ) -> GARCHResult:
        """Fit GARCH(1,1) to a return series via maximum likelihood.

        Parameters
        ----------
        returns:
            Return series (log-returns recommended).
        initial_params:
            Optional ``(omega, alpha, beta)`` starting values.

        Returns
        -------
        GARCHResult
            Fitted parameters and diagnostics.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy is required for GARCH fitting. Install with: pip install scipy")
            raise

        r = returns.dropna().values.astype(np.float64)
        n = len(r)

        if n < 30:
            logger.warning("GARCH: only %d observations — results may be unreliable", n)

        sample_var = float(np.var(r))

        # Initial parameters
        if initial_params is None:
            omega0 = sample_var * 0.05
            alpha0 = 0.08
            beta0 = 0.88
        else:
            omega0, alpha0, beta0 = initial_params

        x0 = np.array([omega0, alpha0, beta0])

        # Bounds: omega > 0, alpha >= 0, beta >= 0
        bounds = [(1e-10, None), (1e-10, 0.9999), (1e-10, 0.9999)]

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if alpha + beta >= 1.0:
                return 1e12

            var_t = np.full(n, sample_var)
            for t in range(1, n):
                var_t[t] = omega + alpha * r[t - 1] ** 2 + beta * var_t[t - 1]
                if var_t[t] <= 0:
                    return 1e12

            # Gaussian log-likelihood (ignoring constant)
            ll = -0.5 * np.sum(np.log(var_t) + r ** 2 / var_t)
            return -ll

        # Stationarity constraint: alpha + beta < 1
        constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

        result = minimize(
            neg_log_likelihood,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        omega, alpha, beta = result.x
        persistence = alpha + beta

        # Long-run variance
        if persistence < 1.0:
            lr_var = omega / (1.0 - persistence)
        else:
            lr_var = sample_var

        # Half-life of shocks
        if persistence > 0 and persistence < 1.0:
            half_life = np.log(2) / np.log(1 / persistence)
        else:
            half_life = float("inf")

        # Compute in-sample conditional variance
        var_series = np.full(n, sample_var)
        for t in range(1, n):
            var_series[t] = omega + alpha * r[t - 1] ** 2 + beta * var_series[t - 1]

        idx = returns.dropna().index
        cond_var = pd.Series(var_series, index=idx, name="garch_variance")

        return GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            log_likelihood=-result.fun,
            long_run_variance=lr_var,
            long_run_volatility=np.sqrt(lr_var * self.ann_factor),
            persistence=persistence,
            half_life=half_life,
            conditional_variance=cond_var,
            converged=result.success,
        )

    def forecast(
        self,
        result: GARCHResult,
        returns: pd.Series,
        horizon: int = 1,
    ) -> dict:
        """Produce h-step-ahead volatility forecasts.

        The GARCH(1,1) h-step forecast is:

            var_{t+h|t} = LR_var + (alpha + beta)^{h-1} * (var_{t+1|t} - LR_var)

        Parameters
        ----------
        result:
            Fitted GARCH result.
        returns:
            The same return series used for fitting.
        horizon:
            Forecast horizon in periods.

        Returns
        -------
        dict
            Keys: ``forecast_variance``, ``forecast_volatility``,
            ``forecast_vol_annualised``, ``ci_lower``, ``ci_upper``,
            ``horizon``.
        """
        r = returns.dropna().values
        omega, alpha, beta = result.omega, result.alpha, result.beta
        persistence = alpha + beta
        lr_var = result.long_run_variance

        # One-step-ahead from last observation
        last_var = float(result.conditional_variance.iloc[-1]) if result.conditional_variance is not None else lr_var
        last_r = float(r[-1])
        one_step_var = omega + alpha * last_r ** 2 + beta * last_var

        # h-step forecast
        if horizon == 1:
            h_var = one_step_var
        else:
            # Cumulative h-step variance
            h_var = lr_var + (persistence ** (horizon - 1)) * (one_step_var - lr_var)

        h_vol = np.sqrt(h_var)
        h_vol_ann = np.sqrt(h_var * self.ann_factor)

        # Approximate confidence intervals (assumes log-normal)
        ci_mult_95 = 1.96
        ci_lower = h_vol_ann * np.exp(-ci_mult_95 * 0.3)  # ~30% vol-of-vol typical
        ci_upper = h_vol_ann * np.exp(ci_mult_95 * 0.3)

        return {
            "forecast_variance": h_var,
            "forecast_volatility": h_vol,
            "forecast_vol_annualised": h_vol_ann,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "horizon": horizon,
        }

    def conditional_volatility(
        self, result: GARCHResult, annualise: bool = True
    ) -> pd.Series:
        """Extract annualised conditional volatility from a fitted result.

        Parameters
        ----------
        result:
            Fitted GARCH result.
        annualise:
            Multiply by sqrt(ann_factor).

        Returns
        -------
        pd.Series
            Conditional volatility series.
        """
        if result.conditional_variance is None:
            raise ValueError("No conditional variance — model has not been fitted.")

        vol = np.sqrt(result.conditional_variance)
        if annualise:
            vol = vol * np.sqrt(self.ann_factor)
        return vol.rename("garch_volatility")
