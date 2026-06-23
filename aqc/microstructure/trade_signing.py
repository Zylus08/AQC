from typing import List, Dict, Any

class TradeSigner:
    """Implements trade signing rules."""
    
    def apply_tick_rule(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Signs trades based on the Tick Rule.
        trades should contain 'price'.
        Returns trades with 'direction' ('BUY'/'SELL').
        """
        signed_trades = []
        last_sign = 'BUY'
        last_price = 0.0
        
        for trade in trades:
            price = trade['price']
            if price > last_price:
                sign = 'BUY'
            elif price < last_price:
                sign = 'SELL'
            else:
                sign = last_sign # Zero tick follows last sign
                
            last_price = price
            last_sign = sign
            
            t = dict(trade)
            t['direction'] = sign
            signed_trades.append(t)
            
        return signed_trades
        
    def apply_lee_ready(self, trades: List[Dict[str, Any]], quotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simplified Lee-Ready algorithm.
        Matches trades to prevailing midprice.
        """
        signed_trades = []
        
        for trade in trades:
            t = dict(trade)
            price = t['price']
            
            # Simple fallback for quotes
            closest_quote = quotes[-1] if quotes else {"mid_price": price}
            mid = closest_quote['mid_price']
            
            if price > mid:
                t['direction'] = 'BUY'
            elif price < mid:
                t['direction'] = 'SELL'
            else:
                t['direction'] = 'UNKNOWN'
                
            signed_trades.append(t)
            
        return signed_trades
