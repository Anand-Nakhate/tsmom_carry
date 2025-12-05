import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_ewma_volatility(returns, lam=0.94):
    # Calculates annualized EWMA volatility using a decay factor lambda
    var = returns.ewm(alpha=(1 - lam), adjust=False).var()
    return np.sqrt(var) * np.sqrt(252)

def calculate_vol_scaled_weights(signal, volatility, target_vol=0.10, vol_floor=0.02, weight_cap=2.0):
    # Computes volatility-scaled weights with floors and caps
    safe_vol = volatility.clip(lower=vol_floor)
    weights = (target_vol / safe_vol) * signal
    return weights.clip(lower=-weight_cap, upper=weight_cap)

def construct_tsmom_portfolio(signals, volatility, target_vol=0.10):
    # Aggregates TSMOM signals across horizons and scales by volatility
    if isinstance(signals, dict):
        combined_signal = pd.DataFrame(0.0, index=list(signals.values())[0].index, columns=list(signals.values())[0].columns)
        count = 0
        for k, v in signals.items():
            combined_signal = combined_signal.add(v, fill_value=0)
            count += 1
        combined_signal = combined_signal / count
    else:
        combined_signal = signals

    return calculate_vol_scaled_weights(combined_signal, volatility, target_vol)

def construct_carry_portfolio(carry_signal, volatility, sleeve_map, target_vol=0.10):
    # Constructs Carry portfolio using asset-level vol scaling
    return calculate_vol_scaled_weights(carry_signal, volatility, target_vol)

def scale_portfolio_weights(weights, returns, target_vol=0.10, window=60):
    naive_ret = (weights.shift(1) * returns).sum(axis=1)
    
    # Estimate realized volatility of this naive portfolio
    rolling_vol = naive_ret.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate scalar
    # Handle zeros and NaNs
    rolling_vol = rolling_vol.replace(0, np.nan).fillna(method='ffill').fillna(target_vol) # Default to no scaling if unknown
    
    scalar = target_vol / rolling_vol
    
    # Cap scalar to avoid extreme leverage (e.g., max 3x leverage boost)
    scalar = scalar.clip(upper=3.0)
    
    scaled_weights = weights.multiply(scalar, axis=0)
    
    return scaled_weights

def stack_portfolios(returns_map, method='equal', window=60, turnover_penalty=0.0):
    # Stacks multiple strategy return streams using various methods
    df = pd.DataFrame(returns_map).dropna()
    
    if method == 'equal':
        weights = pd.Series(1.0 / len(df.columns), index=df.columns)
        return df.dot(weights)
    
    elif method == 'risk_parity':
        vol = df.rolling(window=window).std()
        vol = vol.replace(0, np.nan).fillna(method='ffill')
        inv_vol = 1.0 / vol
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        weights = weights.shift(1)
        return (df * weights).sum(axis=1)
        
    elif method == 'ridge':
        n_assets = len(df.columns)
        results = []
        prev_weights = np.ones(n_assets) / n_assets
        
        for i in range(window, len(df)):
            window_returns = df.iloc[i-window:i]
            cov_matrix = window_returns.cov().values
            
            def objective(w):
                port_var = w.T @ cov_matrix @ w
                ridge_penalty = 0.1 * np.sum(w**2)
                turnover_cost = turnover_penalty * np.sum(np.abs(w - prev_weights))
                return port_var + ridge_penalty + turnover_cost
            
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = tuple((0.0, 1.0) for _ in range(n_assets)) 
            
            try:
                res = minimize(objective, prev_weights, bounds=bounds, constraints=constraints, tol=1e-6)
                if res.success:
                    current_weights = res.x
                else:
                    current_weights = prev_weights
            except Exception:
                current_weights = prev_weights
                
            results.append(current_weights)
            prev_weights = current_weights
            
        weight_df = pd.DataFrame(results, index=df.index[window:], columns=df.columns)
        weight_df = weight_df.shift(1)
        return (df.iloc[window:] * weight_df).sum(axis=1)
        
    else:
        raise ValueError(f"Unknown stacking method: {method}")
