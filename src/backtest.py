import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def align_signals(signals, returns):
    # Aligns signals with returns by lagging signals by one period to avoid look-ahead bias
    aligned_signals = signals.shift(1).dropna()
    common_index = aligned_signals.index.intersection(returns.index)
    return aligned_signals.loc[common_index], returns.loc[common_index]

def calculate_turnover(weights):
    # Calculates portfolio turnover as the average sum of absolute weight changes
    delta = weights.diff().abs().sum(axis=1)
    return delta.mean()

def calculate_transaction_costs(weights, cost_bps=5.0):
    # Estimates transaction costs based on weight changes and a basis point cost
    delta = weights.diff().abs().fillna(0)
    costs = delta.sum(axis=1) * (cost_bps / 10000.0)
    return costs

def run_backtest(weights, returns, cost_bps=5.0):
    # Simulates strategy performance returning gross, net, turnover, and cost metrics
    # Weights are target weights at time t, applied to returns at t+1
    aligned_weights = weights.shift(1).dropna()
    common_idx = aligned_weights.index.intersection(returns.index)
    
    w = aligned_weights.loc[common_idx]
    r = returns.loc[common_idx]
    
    gross_ret = (w * r).sum(axis=1)

    
    costs = calculate_transaction_costs(weights, cost_bps).shift(1).loc[common_idx]
    net_ret = gross_ret - costs
    
    turnover = calculate_turnover(weights)
    
    return pd.DataFrame({
        'Gross': gross_ret,
        'Net': net_ret,
        'Costs': costs
    }), turnover
