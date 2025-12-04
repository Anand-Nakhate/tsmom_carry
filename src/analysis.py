import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

logger = logging.getLogger(__name__)

def calculate_return_stats(returns):
    # Computes comprehensive summary statistics for the return series including higher moments and risk metrics
    stats_df = returns.describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
    stats_df['skew'] = returns.skew()
    stats_df['kurt'] = returns.kurtosis()
    stats_df['pct_up'] = (returns > 0).sum() / returns.count()
    stats_df['ann_vol'] = returns.std() * np.sqrt(252)
    stats_df['sharpe'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    return stats_df

def calculate_autocorrelation(series, lags=20):
    # Calculates the autocorrelation function for the series and its squared values to detect serial dependence and volatility clustering
    s = series.dropna()
    if len(s) < lags:
        return pd.DataFrame()
        
    acf_vals = acf(s, nlags=lags, fft=True)
    sq_acf_vals = acf(s**2, nlags=lags, fft=True)
    
    return pd.DataFrame({
        'lag': range(len(acf_vals)),
        'acf': acf_vals,
        'sq_acf': sq_acf_vals
    }).set_index('lag')

def test_stationarity(series):
    # Performs Augmented Dickey-Fuller and KPSS tests to assess the stationarity of the time series
    s = series.dropna()
    if len(s) < 20 or s.nunique() <= 1:
        return {
            'adf_stat': np.nan, 'adf_p': np.nan,
            'kpss_stat': np.nan, 'kpss_p': np.nan
        }
        
    try:
        adf_res = adfuller(s, autolag='AIC')
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            kpss_res = kpss(s, regression='c', nlags='auto')
        
        return {
            'adf_stat': adf_res[0], 'adf_p': adf_res[1],
            'kpss_stat': kpss_res[0], 'kpss_p': kpss_res[1]
        }
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}")
        return {
            'adf_stat': np.nan, 'adf_p': np.nan,
            'kpss_stat': np.nan, 'kpss_p': np.nan
        }

def calculate_correlations(returns, regime_mask=None):
    # Computes the correlation matrix of returns, optionally filtering by a specific market regime
    if regime_mask is not None:
        return returns.loc[regime_mask].corr()
    return returns.corr()

def calculate_rolling_stats(returns, window=63):
    # Calculates rolling annualized volatility over the specified window
    return returns.rolling(window=window).std() * np.sqrt(252)

def perform_pca(returns, n_components=5):
    # Decomposes the return covariance structure using Principal Component Analysis to identify common factors
    clean_returns = returns.dropna()
    
    if clean_returns.empty:
        return {'explained_variance_ratio': np.array([]), 'components': pd.DataFrame()}
        
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(clean_returns)
    
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    
    components_df = pd.DataFrame(
        pca.components_,
        columns=returns.columns,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': components_df
    }

def check_seasonality(returns):
    # Aggregates average returns by month to identify potential seasonal patterns
    return returns.groupby(returns.index.month).mean()

def detect_outliers(returns, threshold=5.0, method='mad'):
    # Identifies return outliers using robust Median Absolute Deviation (MAD) or standard Z-score methods
    if method == 'mad':
        median = returns.median()
        diff = (returns - median).abs()
        mad = diff.median()
        
        modified_z_scores = 0.6745 * diff / mad
        modified_z_scores = modified_z_scores.replace([np.inf, -np.inf], 0)
        z_scores = modified_z_scores
    else:
        z_scores = (returns - returns.mean()) / returns.std()
        z_scores = z_scores.abs()
    
    outlier_mask = z_scores > threshold
    
    outliers = []
    for col in returns.columns:
        if not outlier_mask[col].any():
            continue
            
        asset_outliers = returns[col][outlier_mask[col]]
        asset_z = z_scores[col][outlier_mask[col]]
        
        for date, ret in asset_outliers.items():
            outliers.append({
                'date': date,
                'asset': col,
                'return': ret,
                'score': asset_z.loc[date],
                'method': method
            })
            
    if not outliers:
        return pd.DataFrame(columns=['date', 'asset', 'return', 'score', 'method'])
        
    return pd.DataFrame(outliers).sort_values('score', ascending=False)

def calculate_max_drawdown(returns):
    # Computes the maximum peak-to-trough decline for each asset's cumulative return path
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    return drawdown.min()

def calculate_var(returns, confidence_level=0.95):
    # Estimates the Historical Value at Risk (VaR) at the specified confidence level
    return -returns.quantile(1 - confidence_level)

def calculate_cvar(returns, confidence_level=0.95):
    # Estimates the Conditional Value at Risk (CVaR) or Expected Shortfall for tail losses beyond VaR
    var = calculate_var(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    if tail_losses.empty:
        return 0.0
    return -tail_losses.mean()

def predictive_regression(signal, returns, horizon=1):
    # Performs predictive regressions of future returns on lagged signals with Newey-West standard errors
    results = []
    
    for asset in signal.columns:
        if asset not in returns.columns:
            continue
            
        X = signal[asset].shift(1)
        y = returns[asset]
        
        if horizon > 1:
            y = y.rolling(horizon).sum().shift(1-horizon)
            
        data = pd.concat([X, y], axis=1).dropna()
        data.columns = ['signal', 'ret']
        
        if len(data) < 60:
            continue
            
        X_reg = sm.add_constant(data['signal'])
        
        model = sm.OLS(data['ret'], X_reg).fit(cov_type='HAC', cov_kwds={'maxlags': horizon})
        
        results.append({
            'asset': asset,
            'alpha': model.params['const'],
            'beta': model.params['signal'],
            't_stat': model.tvalues['signal'],
            'p_value': model.pvalues['signal'],
            'r_squared': model.rsquared
        })
        
    return pd.DataFrame(results).set_index('asset')
