import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_clean(file_path):
    # Loads the master dataset, enforces types, sorts, and removes duplicates
    logger.info(f"Loading dataset from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'expiration' in df.columns:
        df['expiration'] = pd.to_datetime(df['expiration'])
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'settlement', 'open_interest']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    sort_cols = [c for c in ['parent', 'symbol', 'date'] if c in df.columns]
    df = df.sort_values(sort_cols)

    subset_cols = [c for c in ['date', 'instrument_id', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} duplicate rows.")

    if 'root' not in df.columns:
        if 'parent' in df.columns:
            df['root'] = df['parent'].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
        else:
            logger.warning("Neither 'root' nor 'parent' column found. Cannot enrich metadata.")
            return df

    root_meta = {
        '6A': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6B': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6C': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6E': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6J': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6N': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        '6S': {'asset_class': 'FX', 'sleeve': 'FX', 'region': 'G10'},
        'ES': {'asset_class': 'Equity', 'sleeve': 'Equities', 'region': 'US'},
        'NQ': {'asset_class': 'Equity', 'sleeve': 'Equities', 'region': 'US'},
        'RTY':{'asset_class': 'Equity', 'sleeve': 'Equities', 'region': 'US'},
        'NKD':{'asset_class': 'Equity', 'sleeve': 'Equities', 'region': 'Japan'},
        'ZT': {'asset_class': 'Rates', 'sleeve': 'Rates', 'region': 'US'},
        'ZF': {'asset_class': 'Rates', 'sleeve': 'Rates', 'region': 'US'},
        'ZN': {'asset_class': 'Rates', 'sleeve': 'Rates', 'region': 'US'},
        'ZB': {'asset_class': 'Rates', 'sleeve': 'Rates', 'region': 'US'},
        'UB': {'asset_class': 'Rates', 'sleeve': 'Rates', 'region': 'US'},
        'CL': {'asset_class': 'Commodity', 'sleeve': 'Energy', 'region': 'Global'},
        'NG': {'asset_class': 'Commodity', 'sleeve': 'Energy', 'region': 'US'},
        'HO': {'asset_class': 'Commodity', 'sleeve': 'Energy', 'region': 'Global'},
        'RB': {'asset_class': 'Commodity', 'sleeve': 'Energy', 'region': 'Global'},
        'GC': {'asset_class': 'Commodity', 'sleeve': 'Metals', 'region': 'Global'},
        'SI': {'asset_class': 'Commodity', 'sleeve': 'Metals', 'region': 'Global'},
        'HG': {'asset_class': 'Commodity', 'sleeve': 'Metals', 'region': 'Global'},
        'ZC': {'asset_class': 'Commodity', 'sleeve': 'Ags', 'region': 'US'},
        'ZS': {'asset_class': 'Commodity', 'sleeve': 'Ags', 'region': 'US'},
        'ZW': {'asset_class': 'Commodity', 'sleeve': 'Ags', 'region': 'US'},
        'ZL': {'asset_class': 'Commodity', 'sleeve': 'Ags', 'region': 'US'},
        'ZM': {'asset_class': 'Commodity', 'sleeve': 'Ags', 'region': 'US'},
    }

    meta_df = pd.DataFrame(root_meta).T.rename_axis('root').reset_index()
    df = df.merge(meta_df, on='root', how='left')
    
    return df

def run_sanity_checks(df):
    # Runs systematic checks for data integrity including missing values, OHLC consistency, and negative prices
    logger.info("Running sanity checks...")
    issues = {}

    issues['na_counts'] = df.isna().sum().to_dict()

    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        bad_ohlc = df[
            (df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
            (df['low']  > df[['open', 'close', 'high']].min(axis=1))
        ]
        issues['bad_ohlc_rows'] = bad_ohlc.shape[0]
        if not bad_ohlc.empty:
            logger.warning(f"Found {bad_ohlc.shape[0]} rows with inconsistent OHLC.")

    if 'close' in df.columns:
        neg_prices = df[df['close'] < 0]
        issues['neg_price_rows'] = neg_prices.shape[0]
        if not neg_prices.empty:
            logger.warning(f"Found {neg_prices.shape[0]} rows with negative prices.")
            if 'parent' in df.columns:
                issues['neg_price_roots'] = neg_prices['parent'].value_counts().to_dict()

    if 'volume' in df.columns:
        nonpos_vol = df[df['volume'] <= 0]
        issues['nonpos_vol_rows'] = nonpos_vol.shape[0]
        if not nonpos_vol.empty:
            logger.info(f"Found {nonpos_vol.shape[0]} rows with non-positive volume.")

    return issues

def generate_universe_audit(df):
    # Generates a summary of coverage by root including date ranges and volume statistics
    logger.info("Generating universe audit...")
    if 'parent' not in df.columns:
        logger.warning("'parent' column missing, cannot generate audit.")
        return pd.DataFrame()

    agg_dict = {
        'first_date': ('date', 'min'),
        'last_date': ('date', 'max'),
        'n_days': ('date', 'nunique'),
    }
    
    if 'instrument_id' in df.columns:
        agg_dict['avg_daily_contracts'] = ('instrument_id', 'nunique')
    
    if 'volume' in df.columns:
        agg_dict['avg_volume'] = ('volume', 'mean')

    return df.groupby('parent').agg(**agg_dict).sort_index()
