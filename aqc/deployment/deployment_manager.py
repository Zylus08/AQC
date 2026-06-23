"""
aqc/deployment/deployment_manager.py
======================================
Deployment Manager for AQC alphas.

Enforces deployment readiness gates before allocating capital.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase
from aqc.research.tournament.alpha_leaderboard import AlphaLeaderboard

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages the lifecycle transition from research to deployment.

    Parameters
    ----------
    leaderboard:
        AlphaLeaderboard containing the tournament results.
    min_sharpe:
        Minimum Sharpe ratio required for deployment.
    min_ic:
        Minimum Information Coefficient required.
    min_capacity:
        Minimum capacity estimate (in base currency) required.
    """

    def __init__(
        self,
        leaderboard: AlphaLeaderboard,
        min_sharpe: float = 0.5,
        min_ic: float = 0.01,
        min_capacity: float = 50000.0,
    ) -> None:
        self.leaderboard = leaderboard
        self.min_sharpe = min_sharpe
        self.min_ic = min_ic
        self.min_capacity = min_capacity
        self._deployed_alphas: list[str] = []

    def get_deployment_candidates(self) -> list[str]:
        """Get names of alphas that pass all deployment gates."""
        df = self.leaderboard.get_dataframe()
        if df.empty:
            return []

        # Enforce gates
        candidates = df[
            (df["deployment_ready"] == True) &
            (df["sharpe_ratio"] >= self.min_sharpe) &
            (df["information_coefficient"] >= self.min_ic) &
            (df["capacity_estimate"] >= self.min_capacity)
        ]

        logger.info(
            "Deployment Manager found %d candidates out of %d evaluated.",
            len(candidates), len(df),
        )
        return candidates["alpha_name"].tolist()

    def deploy(self, alphas: list[AlphaBase]) -> None:
        """Mark alphas as deployed.

        In a real system, this would register them with the LiveTradingEngine.
        """
        candidates = self.get_deployment_candidates()
        
        for alpha in alphas:
            if alpha.name in candidates:
                logger.info("DEPLOYING: %s", alpha.name)
                # alpha.metadata().status = AlphaStatus.DEPLOYED
                self._deployed_alphas.append(alpha.name)
            else:
                logger.warning("REJECTED: %s did not pass deployment gates.", alpha.name)

    @property
    def deployed_alphas(self) -> list[str]:
        return list(self._deployed_alphas)
