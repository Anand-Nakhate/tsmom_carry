import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_tsmom_signal(prices, k_months):
    # Computes the Time-Series Momentum (TSMOM) signal
    window = 21 * k_months
    k_ret = prices.pct_change(periods=window)
    return np.sign(k_ret)

def calculate_sleeve_neutral_carry(df_carry, meta):
    # Computes the Sleeve-Neutral Carry signal by standardizing carry within each sleeve
    logger.info("Calculating Sleeve-Neutral Carry Signal...")
    
    work_df = df_carry.copy()
    if 'date' not in work_df.columns and 'date' in work_df.index.names:
        work_df = work_df.reset_index()
    if 'root' not in work_df.columns and 'root' in work_df.index.names:
         if 'root' not in work_df.columns: 
             work_df = work_df.reset_index()

    if not all(col in work_df.columns for col in ['date', 'root', 'carry']):
        raise ValueError("Input DataFrame must have 'date', 'root', and 'carry' columns.")

    if 'asset_class' in meta.columns:
        meta_unique = meta[['root', 'asset_class']].drop_duplicates().rename(columns={'asset_class': 'sleeve'})
    elif 'sleeve' in meta.columns:
        meta_unique = meta[['root', 'sleeve']].drop_duplicates()
    else:
        raise ValueError("Meta DataFrame must have 'root' and 'asset_class' (or 'sleeve') columns.")

    merged = work_df.merge(meta_unique, on='root', how='left')
    
    def zscore_func(x):
        if len(x) < 2: return np.nan
        std = x.std(ddof=1)
        if std == 0: return 0.0
        return (x - x.mean()) / std

    merged['carry_z'] = merged.groupby(['date', 'sleeve'])['carry'].transform(zscore_func)
    merged['carry_z'] = merged['carry_z'].clip(lower=-3.0, upper=3.0)
    
    return merged.set_index(['date', 'root'])['carry_z']

def calculate_vol_scaled_signal(signal, vol, target_vol=0.10):
    # Scales the raw signal by inverse volatility
    safe_vol = vol.replace(0, np.nan)
    scaled_sig = (target_vol / safe_vol) * signal
    return scaled_sig.clip(lower=-4.0, upper=4.0)
