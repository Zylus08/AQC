"""
aqc/meta_alpha/alpha_switcher.py
==================================
Dynamically shifts active signals based on the AlphaSelector.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase
from aqc.alpha.alpha_base import AlphaSignal
from aqc.meta_alpha.alpha_selector import AlphaSelector

logger = logging.getLogger(__name__)


class AlphaSwitcher:
    """Acts as a proxy AlphaBase that switches underlying alphas dynamically."""

    def __init__(self, available_alphas: list[AlphaBase]) -> None:
        self.available_alphas = available_alphas
        self.selector = AlphaSelector()

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Route signal generation to the currently optimal alpha."""
        selected_alpha = self.selector.select(data, self.available_alphas)
        
        if selected_alpha is None:
            # Defensive / Flat signal
            return AlphaSignal(
                alpha_name="MetaSwitcher_FLAT",
                timestamp=data.index[-1] if not data.empty else None,
                direction=0,
                strength=0.0,
                confidence=1.0,
                metadata={"regime": "EXTREME", "action": "FLATTEN"}
            )
            
        signal = selected_alpha.generate_signal(data)
        
        # Wrap signal to indicate it was routed
        signal.metadata["routed_by"] = "AlphaSwitcher"
        return signal
