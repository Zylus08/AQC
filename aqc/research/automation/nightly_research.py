"""
aqc/research/automation/nightly_research.py
=============================================
Automated Nightly Research Pipeline.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from aqc.research.tournament.alpha_tournament import AlphaTournament

logger = logging.getLogger(__name__)


class NightlyResearchPipeline:
    """Runs the full alpha evaluation suite."""

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.test_data = test_data

    def run_nightly(self) -> dict[str, Any]:
        """Execute the nightly pipeline.
        
        1. Runs the tournament on all registered alphas.
        2. Generates Phase 10 research answers.
        """
        logger.info("Starting Nightly Research Pipeline.")
        
        tournament = AlphaTournament(self.train_data, self.test_data)
        leaderboard_df = tournament.run()
        
        logger.info("Nightly Tournament complete. Top Alphas:")
        logger.info("\n%s", leaderboard_df.head(3).to_string())
        
        # Answer research questions
        answers = tournament.generate_research_answers()
        
        # Save answers
        import json
        from pathlib import Path
        
        out_dir = Path("reports/automation")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "research_answers.json", "w") as f:
            json.dump(answers, f, indent=4)
            
        logger.info("Saved Research Answers to %s", out_dir / "research_answers.json")
        
        return {
            "leaderboard": leaderboard_df,
            "answers": answers
        }
