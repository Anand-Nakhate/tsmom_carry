import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configuration constants
MIN_DTE = 5
MAX_DTE = 550
MIN_VOL = 10

def compute_days_to_expiry(df):
    # Calculates the number of days between the date and expiration for each contract
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['expiration']):
        df['expiration'] = pd.to_datetime(df['expiration'])
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    exp = df['expiration'].dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(df['expiration']) else df['expiration']
    dt = df['date'].dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(df['date']) else df['date']
    
    df['days_to_expiry'] = (exp.dt.normalize() - dt.dt.normalize()).dt.days
    return df

def select_front_next_contracts(df):
    # Identifies the front and next contracts based on liquidity and time to expiry
    logger.info("Selecting front and next contracts...")
    
    mask = (
        (df['days_to_expiry'] >= MIN_DTE) &
        (df['days_to_expiry'] <= MAX_DTE) &
        (df['volume'] >= MIN_VOL)
    )
    
    candidates = df[mask].copy()
    
    if candidates.empty:
        logger.warning("No valid contract candidates found matching criteria.")
        df['is_front'] = False
        df['is_next'] = False
        return df

    candidates = candidates.sort_values(['root', 'date', 'days_to_expiry'])
    candidates['rank'] = candidates.groupby(['root', 'date']).cumcount()
    
    front_ids = candidates[candidates['rank'] == 0][['root', 'date', 'instrument_id']].rename(columns={'instrument_id': 'front_id'})
    next_ids = candidates[candidates['rank'] == 1][['root', 'date', 'instrument_id']].rename(columns={'instrument_id': 'next_id'})
    
    df_merged = df.merge(front_ids, on=['root', 'date'], how='left')
    df_merged = df_merged.merge(next_ids, on=['root', 'date'], how='left')
    
    df_merged['is_front'] = df_merged['instrument_id'] == df_merged['front_id']
    df_merged['is_next'] = df_merged['instrument_id'] == df_merged['next_id']
    
    return df_merged.drop(columns=['front_id', 'next_id'])

def calculate_instrument_returns(df):
    # Computes daily percentage returns for each instrument
    logger.info("Calculating instrument-level returns...")
    df = df.sort_values(['instrument_id', 'date'])
    df['return'] = df.groupby('instrument_id')['close'].pct_change()
    return df

def calculate_carry_signal(df):
    # Computes the annualized carry signal based on the price differential
    logger.info("Calculating carry signal...")
    
    front = df[df['is_front']].set_index(['root', 'date'])[['close', 'days_to_expiry']]
    next_c = df[df['is_next']].set_index(['root', 'date'])[['close', 'days_to_expiry']]
    
    merged = front.join(next_c, lsuffix='_front', rsuffix='_next', how='inner')
    
    price_diff = merged['close_front'] - merged['close_next']
    time_diff = (merged['days_to_expiry_next'] - merged['days_to_expiry_front'])
    time_diff = time_diff.replace(0, np.nan)
    
    raw_carry = price_diff / merged['close_next']
    annualization_factor = 365 / time_diff
    
    merged['carry'] = raw_carry * annualization_factor
    
    return df.merge(merged[['carry']], left_on=['root', 'date'], right_index=True, how='left')
