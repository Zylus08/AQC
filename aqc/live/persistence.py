"""
aqc/live/persistence.py
=========================
SQLite persistence layer for live trading metrics and snapshots.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PersistenceLayer:
    """SQLite persistence for paper trading state.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "paper_trading.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database tables if they do not exist."""
        cursor = self.conn.cursor()
        
        # Portfolio Snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TEXT PRIMARY KEY,
                total_equity REAL,
                cash REAL,
                gross_exposure REAL,
                net_exposure REAL,
                num_positions INTEGER,
                unrealised_pnl REAL,
                realised_pnl REAL,
                leverage REAL
            )
        ''')
        
        # Signals Audit
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                strategy_id TEXT,
                direction TEXT,
                strength REAL,
                approved BOOLEAN,
                reason TEXT,
                latency_ms REAL
            )
        ''')
        
        # Orders & Fills
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                type TEXT,
                target_qty REAL,
                filled_qty REAL,
                avg_price REAL,
                state TEXT,
                strategy TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')

        # Performance Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT PRIMARY KEY,
                daily_return REAL,
                cagr REAL,
                sharpe REAL,
                sortino REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL
            )
        ''')

        # Health Status
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_status (
                timestamp TEXT PRIMARY KEY,
                state TEXT,
                feed_latency_ms REAL,
                last_heartbeat_s REAL,
                stale_data_warning BOOLEAN,
                rejected_orders_count INTEGER,
                message TEXT
            )
        ''')

        self.conn.commit()

    def save_portfolio_snapshot(self, snap_dict: dict[str, Any]) -> None:
        try:
            pd.DataFrame([snap_dict]).to_sql(
                "portfolio_snapshots",
                self.conn,
                if_exists="append",
                index=False
            )
        except sqlite3.IntegrityError:
            pass # Ignore duplicate timestamp constraint

    def save_signals(self, signals_df: pd.DataFrame) -> None:
        if not signals_df.empty:
            signals_df.to_sql("signal_audit", self.conn, if_exists="append", index=False)

    def save_orders(self, orders_df: pd.DataFrame) -> None:
        if not orders_df.empty:
            # Upsert orders
            orders_df.to_sql("orders", self.conn, if_exists="replace", index=False)

    def save_performance(self, metrics_dict: dict[str, Any]) -> None:
        pd.DataFrame([metrics_dict]).to_sql(
            "performance_metrics", self.conn, if_exists="append", index=False
        )

    def save_health(self, health_dict: dict[str, Any]) -> None:
        pd.DataFrame([health_dict]).to_sql(
            "health_status", self.conn, if_exists="append", index=False
        )

    def load_table(self, table_name: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        except Exception:
            return pd.DataFrame()

    def close(self) -> None:
        self.conn.close()
