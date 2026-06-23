from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

class PaperJournal:
    """Stores persistent logs of trades, signals, forecasts, and regimes."""
    def __init__(self):
        self.trades = []
        self.signals = []
        self.forecasts = []
        self.regimes = []
        self.executions = []
        
    def log_trade(self, trade_data: Dict[str, Any]):
        trade_data['timestamp'] = trade_data.get('timestamp', datetime.now())
        self.trades.append(trade_data)
        
    def log_signal(self, signal_data: Dict[str, Any]):
        signal_data['timestamp'] = signal_data.get('timestamp', datetime.now())
        self.signals.append(signal_data)
        
    def log_forecast(self, forecast_data: Dict[str, Any]):
        forecast_data['timestamp'] = forecast_data.get('timestamp', datetime.now())
        self.forecasts.append(forecast_data)
        
    def log_regime(self, regime_data: Dict[str, Any]):
        regime_data['timestamp'] = regime_data.get('timestamp', datetime.now())
        self.regimes.append(regime_data)
        
    def log_execution(self, execution_data: Dict[str, Any]):
        execution_data['timestamp'] = execution_data.get('timestamp', datetime.now())
        self.executions.append(execution_data)
        
    def generate_daily_report(self) -> Dict[str, Any]:
        return {
            "num_trades": len(self.trades),
            "num_signals": len(self.signals),
            "date": datetime.now().date().isoformat()
        }
