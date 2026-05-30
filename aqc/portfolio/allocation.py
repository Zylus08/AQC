"""
aqc/portfolio/allocation.py
=============================
Portfolio-level Allocation Engine with Constraints.

Computes target weights for multi-asset portfolios and applies
institutional-grade constraints (max weight, leverage, exposure limits).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methodology."""

    FIXED = "fixed"
    VOL_TARGET = "vol_target"
    INVERSE_VOL = "inverse_vol"
    RISK_PARITY = "risk_parity"


@dataclass
class AllocationConstraints:
    """Portfolio allocation constraints.

    Attributes
    ----------
    max_position_weight:
        Maximum weight for a single position (default 0.25 = 25%).
    max_leverage:
        Maximum gross leverage (default 1.0 = no leverage).
    max_gross_exposure:
        Maximum sum of |weights| (default 1.0).
    max_net_exposure:
        Maximum |sum of weights| (default 1.0).
    max_sector_exposure:
        Maximum total weight per sector (default 0.40).
    max_strategy_exposure:
        Maximum weight from a single strategy (default 0.50).
    min_position_weight:
        Minimum weight to maintain a position (default 0.01 = 1%).
    """

    max_position_weight: float = 0.25
    max_leverage: float = 1.0
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_sector_exposure: float = 0.40
    max_strategy_exposure: float = 0.50
    min_position_weight: float = 0.01


@dataclass
class AllocationResult:
    """Portfolio allocation output.

    Attributes
    ----------
    weights:
        Target weights per symbol (dict).
    raw_weights:
        Weights before constraint application.
    method:
        Allocation method used.
    constraints_applied:
        List of constraints that were binding.
    gross_exposure:
        Sum of |weights|.
    net_exposure:
        Sum of weights.
    """

    weights: dict[str, float]
    raw_weights: dict[str, float]
    method: AllocationMethod
    constraints_applied: list[str]
    gross_exposure: float = 0.0
    net_exposure: float = 0.0


class PortfolioAllocator:
    """Multi-asset allocation engine with constraint enforcement.

    Parameters
    ----------
    constraints:
        Allocation constraints.
    ann_factor:
        Annualisation factor (default 252).

    Examples
    --------
    >>> allocator = PortfolioAllocator()
    >>> result = allocator.allocate(
    ...     symbols=["AAPL", "MSFT", "TLT"],
    ...     vols={"AAPL": 0.25, "MSFT": 0.22, "TLT": 0.10},
    ...     method=AllocationMethod.INVERSE_VOL,
    ... )
    """

    def __init__(
        self,
        constraints: Optional[AllocationConstraints] = None,
        ann_factor: int = 252,
    ) -> None:
        self.constraints = constraints or AllocationConstraints()
        self.ann_factor = ann_factor
        self._allocation_history: list[AllocationResult] = []

    def allocate(
        self,
        symbols: list[str],
        vols: Optional[dict[str, float]] = None,
        signals: Optional[dict[str, float]] = None,
        method: AllocationMethod = AllocationMethod.INVERSE_VOL,
        target_vol: float = 0.10,
        sectors: Optional[dict[str, str]] = None,
        strategies: Optional[dict[str, str]] = None,
    ) -> AllocationResult:
        """Compute constrained portfolio weights.

        Parameters
        ----------
        symbols:
            Asset tickers.
        vols:
            Forecast volatility per symbol (required for vol-based methods).
        signals:
            Signal strength per symbol (for fixed sizing).
        method:
            Allocation methodology.
        target_vol:
            Target portfolio volatility (for vol-target method).
        sectors:
            Symbol → sector mapping (for sector constraint).
        strategies:
            Symbol → strategy mapping (for strategy constraint).

        Returns
        -------
        AllocationResult
        """
        if method == AllocationMethod.FIXED:
            raw = self._fixed_weights(symbols, signals)
        elif method == AllocationMethod.INVERSE_VOL:
            raw = self._inverse_vol_weights(symbols, vols or {})
        elif method == AllocationMethod.RISK_PARITY:
            raw = self._risk_parity_weights(symbols, vols or {}, target_vol)
        elif method == AllocationMethod.VOL_TARGET:
            raw = self._vol_target_weights(symbols, vols or {}, target_vol)
        else:
            raw = self._fixed_weights(symbols, signals)

        # Apply constraints
        constrained, applied = self._apply_constraints(raw, sectors, strategies)

        result = AllocationResult(
            weights=constrained,
            raw_weights=raw,
            method=method,
            constraints_applied=applied,
            gross_exposure=round(sum(abs(w) for w in constrained.values()), 6),
            net_exposure=round(sum(constrained.values()), 6),
        )

        self._allocation_history.append(result)
        return result

    def get_allocation_history(self) -> list[AllocationResult]:
        """Return all past allocation results."""
        return self._allocation_history

    # ------------------------------------------------------------------
    # Weight computation methods
    # ------------------------------------------------------------------

    def _fixed_weights(
        self,
        symbols: list[str],
        signals: Optional[dict[str, float]],
    ) -> dict[str, float]:
        """Equal-weight or signal-proportional."""
        n = len(symbols)
        if not symbols:
            return {}

        if signals:
            total = sum(abs(v) for v in signals.values()) or 1.0
            return {s: round(signals.get(s, 0) / total, 6) for s in symbols}
        else:
            w = round(1.0 / n, 6)
            return {s: w for s in symbols}

    def _inverse_vol_weights(
        self,
        symbols: list[str],
        vols: dict[str, float],
    ) -> dict[str, float]:
        """Weight inversely proportional to volatility."""
        valid = {s: v for s, v in vols.items() if v > 0 and s in symbols}
        if not valid:
            return self._fixed_weights(symbols, None)

        inv = {s: 1.0 / v for s, v in valid.items()}
        total = sum(inv.values())

        weights = {}
        for s in symbols:
            if s in inv:
                weights[s] = round(inv[s] / total, 6)
            else:
                weights[s] = 0.0
        return weights

    def _risk_parity_weights(
        self,
        symbols: list[str],
        vols: dict[str, float],
        target_vol: float,
    ) -> dict[str, float]:
        """Equal risk contribution per asset.

        Simplified: weight_i = (1/vol_i) / sum(1/vol_j), scaled to target vol.
        """
        # In the simplified case (no correlations), risk parity = inverse vol
        return self._inverse_vol_weights(symbols, vols)

    def _vol_target_weights(
        self,
        symbols: list[str],
        vols: dict[str, float],
        target_vol: float,
    ) -> dict[str, float]:
        """Target a specific portfolio volatility.

        weight_i = target_vol / (N * vol_i)  [simplified, ignoring correlations]
        """
        n = len(symbols)
        if n == 0:
            return {}

        weights = {}
        for s in symbols:
            v = vols.get(s, 0)
            if v > 0:
                weights[s] = round(target_vol / (n * v), 6)
            else:
                weights[s] = round(1.0 / n, 6)
        return weights

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def _apply_constraints(
        self,
        weights: dict[str, float],
        sectors: Optional[dict[str, str]],
        strategies: Optional[dict[str, str]],
    ) -> tuple[dict[str, float], list[str]]:
        """Apply allocation constraints, returning (constrained_weights, applied_constraints)."""
        w = dict(weights)
        applied = []

        # 1. Max position weight
        for s in list(w):
            if abs(w[s]) > self.constraints.max_position_weight:
                w[s] = np.sign(w[s]) * self.constraints.max_position_weight
                applied.append(f"max_position_weight:{s}")

        # 2. Min position weight (eliminate dust positions)
        for s in list(w):
            if 0 < abs(w[s]) < self.constraints.min_position_weight:
                w[s] = 0.0
                applied.append(f"min_position_weight:{s}")

        # 3. Max sector exposure
        if sectors:
            sector_weights: dict[str, float] = {}
            for s, weight in w.items():
                sector = sectors.get(s, "unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)

            for sector, total in sector_weights.items():
                if total > self.constraints.max_sector_exposure:
                    scale = self.constraints.max_sector_exposure / total
                    for s in w:
                        if sectors.get(s) == sector:
                            w[s] *= scale
                    applied.append(f"max_sector_exposure:{sector}")

        # 4. Max strategy exposure
        if strategies:
            strat_weights: dict[str, float] = {}
            for s, weight in w.items():
                strat = strategies.get(s, "unknown")
                strat_weights[strat] = strat_weights.get(strat, 0) + abs(weight)

            for strat, total in strat_weights.items():
                if total > self.constraints.max_strategy_exposure:
                    scale = self.constraints.max_strategy_exposure / total
                    for s in w:
                        if strategies.get(s) == strat:
                            w[s] *= scale
                    applied.append(f"max_strategy_exposure:{strat}")

        # 5. Max gross exposure (rescale all proportionally)
        gross = sum(abs(v) for v in w.values())
        if gross > self.constraints.max_gross_exposure:
            scale = self.constraints.max_gross_exposure / gross
            w = {s: round(v * scale, 6) for s, v in w.items()}
            applied.append("max_gross_exposure")

        # 6. Max net exposure
        net = sum(w.values())
        if abs(net) > self.constraints.max_net_exposure:
            # Proportionally reduce
            if abs(net) > 0:
                scale = self.constraints.max_net_exposure / abs(net)
                w = {s: round(v * scale, 6) for s, v in w.items()}
                applied.append("max_net_exposure")

        # 7. Max leverage
        gross_final = sum(abs(v) for v in w.values())
        if gross_final > self.constraints.max_leverage:
            scale = self.constraints.max_leverage / gross_final
            w = {s: round(v * scale, 6) for s, v in w.items()}
            applied.append("max_leverage")

        return w, applied
