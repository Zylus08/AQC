"""
aqc/research/hypothesis_lab/hypothesis_registry.py
====================================================
Persistent registry for Alpha Hypotheses.

Uses a local JSON file to persist hypothesis state across sessions.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from aqc.research.hypothesis_lab.hypothesis import AlphaHypothesis, HypothesisStatus

logger = logging.getLogger(__name__)


class HypothesisRegistry:
    """Manages persistent storage of alpha hypotheses."""

    def __init__(self, filepath: str = "data/research/hypotheses.json") -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._hypotheses: dict[str, AlphaHypothesis] = {}
        self.load()

    def load(self) -> None:
        """Load hypotheses from disk."""
        if not self.filepath.exists():
            logger.info("No existing hypothesis registry found at %s", self.filepath)
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for hid, hdata in data.items():
                self._hypotheses[hid] = AlphaHypothesis.from_dict(hdata)
                
            logger.info("Loaded %d hypotheses from registry.", len(self._hypotheses))
        except Exception as e:
            logger.error("Failed to load hypotheses: %s", e)

    def save(self) -> None:
        """Save hypotheses to disk."""
        try:
            data = {hid: h.to_dict() for hid, h in self._hypotheses.items()}
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.debug("Saved %d hypotheses to registry.", len(self._hypotheses))
        except Exception as e:
            logger.error("Failed to save hypotheses: %s", e)

    def register(self, hypothesis: AlphaHypothesis) -> None:
        """Add a new hypothesis or update an existing one."""
        self._hypotheses[hypothesis.id] = hypothesis
        self.save()
        logger.info("Registered hypothesis: %s", hypothesis.id)

    def get(self, hypothesis_id: str) -> Optional[AlphaHypothesis]:
        """Retrieve a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def get_all(self) -> list[AlphaHypothesis]:
        """Get all registered hypotheses."""
        return list(self._hypotheses.values())

    def get_by_status(self, status: HypothesisStatus) -> list[AlphaHypothesis]:
        """Get hypotheses filtered by status."""
        return [h for h in self._hypotheses.values() if h.status == status]
