"""
aqc/research/hypothesis_lab/hypothesis.py
===========================================
Alpha Hypothesis data structures.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Lifecycle status of a research hypothesis."""

    IDEA = "IDEA"
    TESTING = "TESTING"
    FAILED = "FAILED"
    PROMISING = "PROMISING"
    DEPLOYED = "DEPLOYED"
    RETIRED = "RETIRED"


@dataclass
class AlphaHypothesis:
    """Represents an alpha idea from inception to deployment.

    Attributes
    ----------
    id:
        Unique identifier for the hypothesis.
    title:
        Short descriptive name.
    creator:
        Author or researcher name.
    description:
        Detailed explanation of the alpha mechanism.
    expected_mechanism:
        The fundamental or microstructure reason why this should work.
    feature_set:
        List of features required to test this hypothesis.
    status:
        Current stage in the research lifecycle.
    test_results:
        Dictionary of baseline evaluation metrics.
    created_at:
        Timestamp of idea creation.
    updated_at:
        Timestamp of last status update.
    notes:
        Any qualitative notes or findings.
    """

    id: str
    title: str
    creator: str
    description: str
    expected_mechanism: str
    feature_set: list[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.IDEA
    test_results: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    notes: list[str] = field(default_factory=list)

    def update_status(self, new_status: HypothesisStatus, note: str = "") -> None:
        """Update the status and log a note."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
        if note:
            self.add_note(f"Status changed to {new_status.value}: {note}")

    def add_note(self, note: str) -> None:
        """Add a qualitative note."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.notes.append(f"[{timestamp}] {note}")
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "creator": self.creator,
            "description": self.description,
            "expected_mechanism": self.expected_mechanism,
            "feature_set": self.feature_set,
            "status": self.status.value,
            "test_results": self.test_results,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlphaHypothesis":
        # Handle datetime parsing if they are strings
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
            
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
            
        return cls(
            id=data["id"],
            title=data["title"],
            creator=data["creator"],
            description=data["description"],
            expected_mechanism=data["expected_mechanism"],
            feature_set=data.get("feature_set", []),
            status=HypothesisStatus(data.get("status", "IDEA")),
            test_results=data.get("test_results", {}),
            created_at=created or datetime.utcnow(),
            updated_at=updated or datetime.utcnow(),
            notes=data.get("notes", []),
        )
