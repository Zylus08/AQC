"""
aqc/research/alpha_decay/alpha_lifecycle.py
=============================================
Manages the end-of-life process for decayed alphas.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.research.hypothesis_lab.hypothesis_registry import HypothesisRegistry
from aqc.research.hypothesis_lab.hypothesis import HypothesisStatus

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages transitions into the RETIRED state."""

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry

    def retire_alpha(self, hypothesis_id: str, reason: str) -> bool:
        """Mark a hypothesis/alpha as retired.

        Returns
        -------
        bool
            True if successfully retired.
        """
        hyp = self.registry.get(hypothesis_id)
        if not hyp:
            logger.error("Cannot retire. Hypothesis %s not found.", hypothesis_id)
            return False

        if hyp.status == HypothesisStatus.RETIRED:
            logger.info("%s is already retired.", hypothesis_id)
            return True

        logger.warning("RETIRING ALPHA: %s. Reason: %s", hypothesis_id, reason)
        hyp.update_status(HypothesisStatus.RETIRED, f"Retired due to decay: {reason}")
        self.registry.save()
        
        # In a live system, this would also trigger the StrategyRouter/KillSwitch 
        # to pull all capital from this alpha.
        return True
