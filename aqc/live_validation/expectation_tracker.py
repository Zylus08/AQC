from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

@dataclass
class ExpectedPerformanceProfile:
    """Stores expected behaviour from backtests, walk-forward studies, or paper trading."""
    sharpe: float
    sortino: float
    cagr: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    signal_frequency: float  # signals per month/day depending on resolution
    forecast_accuracy: float # e.g. RMSE
    
    # Confidence intervals (lower, upper)
    sharpe_range: Tuple[float, float]
    win_rate_range: Tuple[float, float]
    profit_factor_range: Tuple[float, float]
    signal_frequency_range: Tuple[float, float]
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpectationTracker:
    """Manages the expected profiles for different strategies/models."""
    def __init__(self):
        self._profiles: Dict[str, ExpectedPerformanceProfile] = {}
        
    def add_profile(self, strategy_id: str, profile: ExpectedPerformanceProfile) -> None:
        """Adds or updates an expectation profile for a strategy."""
        self._profiles[strategy_id] = profile
        
    def get_profile(self, strategy_id: str) -> ExpectedPerformanceProfile:
        """Retrieves the expectation profile for a strategy."""
        if strategy_id not in self._profiles:
            raise KeyError(f"No expectation profile found for strategy: {strategy_id}")
        return self._profiles[strategy_id]
