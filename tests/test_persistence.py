"""
tests/test_persistence.py
===========================
Tests for SQLite Persistence Layer.

Author: Saksham Mishra — AlgoQuant Club
"""
import os
import pandas as pd
from aqc.live.persistence import PersistenceLayer


def test_persistence_layer():
    db_path = "test_persistence.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = PersistenceLayer(db_path)
    
    snap = {
        "timestamp": str(pd.Timestamp.utcnow()),
        "total_equity": 100000.0,
        "cash": 50000.0,
        "gross_exposure": 50000.0,
        "net_exposure": 50000.0,
        "num_positions": 2,
        "unrealised_pnl": 500.0,
        "realised_pnl": 0.0,
        "leverage": 0.5
    }
    
    db.save_portfolio_snapshot(snap)
    
    df = db.load_table("portfolio_snapshots")
    assert not df.empty
    assert df.iloc[0]["total_equity"] == 100000.0
    
    db.close()
    
    if os.path.exists(db_path):
        os.remove(db_path)
