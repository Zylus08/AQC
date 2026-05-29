"""Strategy framework — base class and sample implementations."""
from aqc.strategies.base_strategy import BaseStrategy
from aqc.strategies.sample_strategy import SMACrossoverStrategy

__all__ = ["BaseStrategy", "SMACrossoverStrategy"]
