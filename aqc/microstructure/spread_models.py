class SpreadModels:
    """Models spread components (adverse selection, inventory holding, order processing)."""
    
    def decompose_spread(self, spread: float, adverse_selection_ratio: float = 0.4, inventory_ratio: float = 0.4) -> dict:
        return {
            "total_spread": spread,
            "adverse_selection_component": spread * adverse_selection_ratio,
            "inventory_component": spread * inventory_ratio,
            "order_processing_component": spread * (1 - adverse_selection_ratio - inventory_ratio)
        }
