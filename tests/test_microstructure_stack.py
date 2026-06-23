import pytest
import numpy as np
import pandas as pd
from aqc.institutional.bayesian_validator import BayesianAlphaValidator
from aqc.institutional.shadow_backtester import ShadowBacktester
from aqc.institutional.alpha_reality_check import AlphaRealityCheck
from aqc.microstructure.trade_signing import TradeSigner
from aqc.microstructure.flow_toxicity import FlowToxicity
from aqc.orderbook.imbalance_engine import ImbalanceEngine
from aqc.orderbook.microprice import MicropriceEstimator
from aqc.execution.market_impact_model import MarketImpactModel
from aqc.execution.execution_optimizer import ExecutionOptimizer

def test_bayesian_validator():
    validator = BayesianAlphaValidator(prior_mu=1.0, prior_std=0.5)
    validator.update(observed_mean=0.5, observed_std=0.2, n_samples=30)
    probs = validator.get_probabilities(zero_threshold=0.0)
    assert 'p_alive' in probs
    assert probs['posterior_mu'] < 1.0
    
def test_shadow_backtester():
    sb = ShadowBacktester()
    exp = [{'symbol': 'AAPL', 'direction': 'BUY', 'price': 150.0}]
    act = [{'symbol': 'AAPL', 'direction': 'BUY', 'price': 151.0}]
    res = sb.compare_trades(exp, act)
    assert res['missed_trades'] == 0
    assert res['avg_price_divergence'] > 0
    
def test_alpha_reality_check():
    arc = AlphaRealityCheck()
    res = arc.check_survival(15.0, 1.0, 0.5, 1.2, 0.8, 0.5)
    assert res['net_alpha'] == 11.0
    assert res['survives'] is True
    
def test_trade_signer():
    ts = TradeSigner()
    trades = [{'price': 100}, {'price': 101}, {'price': 101}, {'price': 99}]
    signed = ts.apply_tick_rule(trades)
    assert signed[0]['direction'] == 'BUY' # default
    assert signed[1]['direction'] == 'BUY'
    assert signed[2]['direction'] == 'BUY' # zero tick follows last
    assert signed[3]['direction'] == 'SELL'
    
def test_flow_toxicity():
    ft = FlowToxicity()
    buckets = [{'buy_vol': 100, 'sell_vol': 50}, {'buy_vol': 20, 'sell_vol': 150}]
    vpin = ft.calculate_vpin(buckets, 150) # total vol = 300
    # imbalance = 50 + 130 = 180
    # vpin = 180 / 300 = 0.6
    assert abs(vpin - 0.6) < 1e-5
    
def test_imbalance_engine():
    ie = ImbalanceEngine()
    bids = [(100, 50), (99, 50)]
    asks = [(101, 100), (102, 100)]
    imb = ie.compute_imbalance(bids, asks, 1)
    # bids vol = 50, asks vol = 100
    # imb = (50 - 100) / 150 = -1/3
    assert abs(imb - (-0.33333333333)) < 1e-5
    
def test_microprice():
    me = MicropriceEstimator()
    mp = me.calculate_microprice(100, 50, 101, 100)
    # mp = (100*100 + 101*50) / 150 = (10000 + 5050)/150 = 15050/150 = 100.333
    assert abs(mp - 100.33333333) < 1e-5
    
def test_market_impact():
    mim = MarketImpactModel()
    res = mim.estimate_impact(1000, 100000, 0.02)
    assert res['temporary_bps'] > 0
    assert res['permanent_bps'] > 0
    
def test_execution_optimizer():
    opt = ExecutionOptimizer()
    res = opt.optimize_execution(15000, 100000)
    # participation = 0.15 > 0.10 -> VWAP
    assert res['best_strategy'] == 'VWAP'
