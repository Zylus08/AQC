class QueuePosition:
    """Estimates queue position for limit orders."""
    
    def estimate_position(self, order_price: float, side: str, bids: list, asks: list) -> float:
        """
        Returns estimated volume ahead in the queue.
        """
        volume_ahead = 0.0
        if side == "BUY":
            for p, v in bids:
                if p > order_price:
                    volume_ahead += v
                elif p == order_price:
                    volume_ahead += v # Assume we are at the back of the queue
                    break
        else:
            for p, v in asks:
                if p < order_price:
                    volume_ahead += v
                elif p == order_price:
                    volume_ahead += v
                    break
        return volume_ahead
