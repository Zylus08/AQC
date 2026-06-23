"""
aqc/research/tournament/
========================
Alpha Tournament Engine.

Runs all registered alphas through backtesting, walk-forward, 
and reality checks to generate a ranked leaderboard.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.tournament.alpha_tournament import AlphaTournament
from aqc.research.tournament.alpha_leaderboard import AlphaLeaderboard
from aqc.research.tournament.alpha_comparison import AlphaComparator

__all__ = [
    "AlphaTournament",
    "AlphaLeaderboard",
    "AlphaComparator",
]
