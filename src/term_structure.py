import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration constants
MIN_DTE = 5
MAX_DTE = 550
MIN_VOL = 10

def compute_days_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of days between the date and expiration for each contract.
    
    Args:
        df: DataFrame containing 'date' and 'expiration' columns.
        
    Returns:
        DataFrame with an added 'days_to_expiry' column.
    """
    df = df.copy()
    
    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df['expiration']):
        df['expiration'] = pd.to_datetime(df['expiration'])
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    # Remove timezone information for calculation
    exp = df['expiration'].dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(df['expiration']) else df['expiration']
    dt = df['date'].dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(df['date']) else df['date']
    
    # Calculate days to expiry
    df['days_to_expiry'] = (exp.dt.normalize() - dt.dt.normalize()).dt.days
    return df

def select_front_next_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the front and next contracts based on liquidity and time to expiry.
    Uses vectorized operations for performance.
    
    Args:
        df: DataFrame containing 'root', 'date', 'days_to_expiry', 'volume', and 'instrument_id'.
        
    Returns:
        DataFrame with 'is_front' and 'is_next' boolean columns.
    """
    logger.info("Selecting front and next contracts...")
    
    # Filter candidates based on criteria
    mask = (
        (df['days_to_expiry'] >= MIN_DTE) &
        (df['days_to_expiry'] <= MAX_DTE) &
        (df['volume'] >= MIN_VOL)
    )
    
    # Create a subset of valid candidates
    candidates = df[mask].copy()
    
    if candidates.empty:
        logger.warning("No valid contract candidates found matching criteria.")
        df['is_front'] = False
        df['is_next'] = False
        return df

    # Sort by root, date, and days_to_expiry (ascending)
    # This ensures the contract with the smallest valid DTE comes first
    candidates = candidates.sort_values(['root', 'date', 'days_to_expiry'])
    
    # Assign ranks within each (root, date) group
    # Rank 0 = Front, Rank 1 = Next
    candidates['rank'] = candidates.groupby(['root', 'date']).cumcount()
    
    # Extract front and next IDs
    front_ids = candidates[candidates['rank'] == 0][['root', 'date', 'instrument_id']].rename(columns={'instrument_id': 'front_id'})
    next_ids = candidates[candidates['rank'] == 1][['root', 'date', 'instrument_id']].rename(columns={'instrument_id': 'next_id'})
    
    # Merge IDs back to the original dataframe
    # We use left joins to keep all original rows
    df_merged = df.merge(front_ids, on=['root', 'date'], how='left')
    df_merged = df_merged.merge(next_ids, on=['root', 'date'], how='left')
    
    # Create boolean flags
    df_merged['is_front'] = df_merged['instrument_id'] == df_merged['front_id']
    df_merged['is_next'] = df_merged['instrument_id'] == df_merged['next_id']
    
    # Cleanup temporary columns
    return df_merged.drop(columns=['front_id', 'next_id'])

def calculate_instrument_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily percentage returns for each instrument.
    
    Args:
        df: DataFrame containing 'instrument_id', 'date', and 'close'.
        
    Returns:
        DataFrame with an added 'return' column.
    """
    logger.info("Calculating instrument-level returns...")
    df = df.sort_values(['instrument_id', 'date'])
    # Calculate percentage change per instrument
    df['return'] = df.groupby('instrument_id')['close'].pct_change()
    return df

def calculate_carry_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the annualized carry signal based on the price differential between front and next contracts.
    Carry = (Price_Front - Price_Next) / Price_Next * (365 / Time_Diff)
    
    Args:
        df: DataFrame with 'is_front', 'is_next', 'root', 'date', 'close', 'days_to_expiry'.
        
    Returns:
        DataFrame with an added 'carry' column.
    """
    logger.info("Calculating carry signal...")
    
    # Extract front and next contract data
    # We enforce 1-to-1 mapping by ensuring no duplicates for root/date in the subsets
    front = df[df['is_front']].set_index(['root', 'date'])[['close', 'days_to_expiry']]
    next_c = df[df['is_next']].set_index(['root', 'date'])[['close', 'days_to_expiry']]
    
    # Join front and next data on root/date
    merged = front.join(next_c, lsuffix='_front', rsuffix='_next', how='inner')
    
    # Calculate differentials
    price_diff = merged['close_front'] - merged['close_next']
    time_diff = (merged['days_to_expiry_next'] - merged['days_to_expiry_front'])
    
    # Avoid division by zero in time_diff
    time_diff = time_diff.replace(0, np.nan)
    
    # Calculate annualized carry
    # We normalize by the next contract price (the one we would roll into)
    raw_carry = price_diff / merged['close_next']
    annualization_factor = 365 / time_diff
    
    merged['carry'] = raw_carry * annualization_factor
    
    # Merge carry signal back to the original dataframe
    # The signal is broadcasted to all rows for the same root/date
    return df.merge(merged[['carry']], left_on=['root', 'date'], right_index=True, how='left')
