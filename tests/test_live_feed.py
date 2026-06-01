"""
tests/test_live_feed.py
=========================
Tests for Live Data Feeds.

Author: Saksham Mishra — AlgoQuant Club
"""
import asyncio
import pandas as pd
import pytest

from aqc.live.live_data_feed import SimulatedFeed

@pytest.mark.asyncio
async def test_simulated_feed():
    idx = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0],
    }, index=idx)
    
    feed = SimulatedFeed({"AAPL": df}, interval_seconds=0.01)
    connected = await feed.connect()
    assert connected
    assert feed.is_connected
    
    events = []
    async for event in feed.stream_bars():
        events.append(event)
        
    assert len(events) == 3
    assert events[0].close_price == 100.0
    assert events[-1].close_price == 102.0
    
    await feed.disconnect()
    assert not feed.is_connected
