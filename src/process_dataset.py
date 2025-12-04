"""
Master Dataset Processor.
Loads raw Databento files, processes them, and creates the master dataset.
"""
import os
import logging
import functools
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import databento as db

from src.config import GLBX_UNIVERSE, RootContract

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("./databento_data")
OUTPUT_FILE = "master_dataset.csv"

def _load_single_file(file_path: Path, add_date_from_filename: bool = False) -> pd.DataFrame:
    """Helper function to load a single DBN file, intended for parallel execution."""
    try:
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()
        
        if add_date_from_filename:
            # Extract date from filename: glbx-mdp3-YYYYMMDD.schema.dbn.zst
            try:
                parts = file_path.name.split('.')
                date_str = None
                for p in parts:
                    if '-' in p:
                        subparts = p.split('-')
                        for sp in subparts:
                            if len(sp) == 8 and sp.isdigit():
                                date_str = sp
                                break
                    if date_str: break
                
                if date_str:
                    date_val = pd.to_datetime(date_str).date()
                    df['date'] = date_val
            except Exception as e:
                # Use print here as this runs in a subprocess where logger might not be configured
                print(f"      -> [WARN] Failed to extract date from {file_path.name}: {e}")
        return df
    except Exception as e:
        print(f"      -> [ERROR] Failed to read {file_path.name}: {e}")
        return pd.DataFrame()

class DatasetProcessor:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        # Ensure log directory exists
        Path("logs").mkdir(exist_ok=True)

    def load_data_from_dir(self, directory: Path, limit: int = None, add_date_from_filename: bool = False) -> pd.DataFrame:
        logger.info(f"Loading from {directory}...")
        if directory.is_file():
            return db.DBNStore.from_file(directory).to_df()
            
        files = sorted(list(directory.glob("*.dbn.zst")))
        if not files:
            logger.warning(f"No .dbn.zst files found in {directory}")
            return pd.DataFrame()
        
        if limit:
            files = files[:limit]
        
        logger.info(f"Starting parallel load of {len(files)} files...")
        max_workers = os.cpu_count() or 4
        
        dfs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            load_func = functools.partial(_load_single_file, add_date_from_filename=add_date_from_filename)
            results = list(executor.map(load_func, files))
            dfs = [df for df in results if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
            
        logger.info(f"Concatenating {len(dfs)} DataFrames...")
        return pd.concat(dfs, ignore_index=True)

    def process_data(self, 
                     file_paths: Dict[str, Path], 
                     universe: List[RootContract],
                     limit: int = None) -> pd.DataFrame:
        """
        Reads DBN files from directories and merges them into a single panel.
        """
        logger.info("Processing local DBN files...")

        # --- A. Load Definitions (Metadata) ---
        logger.info("Loading Definitions...")
        df_def = self.load_data_from_dir(file_paths["definition"], limit=limit)
        
        if 'instrument_class' in df_def.columns:
            df_def = df_def[df_def['instrument_class'] == 'F'] # Futures only

        # De-dupe: Keep last definition per instrument
        if not df_def.empty:
            df_def = df_def.sort_values('ts_event').groupby('instrument_id').tail(1)
        
        # Select meta cols
        meta_cols = ['instrument_id', 'raw_symbol', 'expiration', 'min_price_increment', 'currency']
        meta_cols = [c for c in meta_cols if c in df_def.columns]
        df_meta = df_def[meta_cols].copy()

        # --- B. Load OHLCV ---
        logger.info("Loading OHLCV...")
        df_ohlcv = self.load_data_from_dir(file_paths["ohlcv-1d"], limit=limit, add_date_from_filename=True)
        
        # Standardize Date
        if not df_ohlcv.empty:
            if 'date' not in df_ohlcv.columns:
                 if 'ts_event' in df_ohlcv.columns:
                    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['ts_event']).dt.date
                 else:
                    logger.warning("'date' and 'ts_event' missing in OHLCV.")
        
        # --- C. Load Statistics (Settlement & OI) ---
        logger.info("Loading Statistics...")
        df_stats = self.load_data_from_dir(file_paths["statistics"], limit=limit, add_date_from_filename=True)
        
        if not df_stats.empty:
            if 'date' not in df_stats.columns and 'ts_event' in df_stats.columns:
                df_stats['date'] = pd.to_datetime(df_stats['ts_event']).dt.date

            # Filter for Settlement(6), OI(5), Vol(4)
            mask = df_stats['stat_type'].isin([
                db.StatType.SETTLEMENT_PRICE,
                db.StatType.OPEN_INTEREST,
                db.StatType.CLEARED_VOLUME
            ])
            df_stats = df_stats[mask].copy()
            
            # Take last update per day
            df_stats = df_stats.sort_values('ts_event').groupby(['instrument_id', 'date', 'stat_type']).tail(1)

            # Pivot
            if not df_stats.empty:
                df_stats_pivoted = df_stats.pivot(
                    index=['instrument_id', 'date'], 
                    columns='stat_type', 
                    values=['price', 'quantity']
                )
                
                # Robust Column Extraction
                try:
                    settle = df_stats_pivoted[('price', db.StatType.SETTLEMENT_PRICE)]
                except KeyError:
                    settle = pd.Series(dtype=float)

                try:
                    oi = df_stats_pivoted[('quantity', db.StatType.OPEN_INTEREST)]
                except KeyError:
                    oi = pd.Series(dtype=float)

                try:
                    vol = df_stats_pivoted[('quantity', db.StatType.CLEARED_VOLUME)]
                except KeyError:
                    vol = pd.Series(dtype=float)

                df_stats_clean = pd.DataFrame({
                    'settlement': settle,
                    'open_interest': oi,
                    'cleared_volume_stat': vol
                }).reset_index()
            else:
                 df_stats_clean = pd.DataFrame(columns=['instrument_id', 'date', 'settlement', 'open_interest', 'cleared_volume_stat'])
        else:
            df_stats_clean = pd.DataFrame(columns=['instrument_id', 'date', 'settlement', 'open_interest', 'cleared_volume_stat'])
            
        # Ensure columns exist
        for col in ['instrument_id', 'date']:
            if col not in df_stats_clean.columns:
                df_stats_clean[col] = pd.Series(dtype='object')

        # --- D. Merge All ---
        logger.info("Merging...")
        full_df = pd.merge(df_ohlcv, df_meta, on='instrument_id', how='left')
        full_df = pd.merge(full_df, df_stats_clean, on=['instrument_id', 'date'], how='left')
        
        # --- E. Enriched Metadata (Asset Class, Region) ---
        logger.info("Mapping Asset Classes...")
        root_lookup = {}
        for r in universe:
            code = r.parent.split('.')[0] 
            root_lookup[code] = r

        def get_root_info(raw_symbol):
            if not isinstance(raw_symbol, str): return None, None, None
            for code, r in root_lookup.items():
                if raw_symbol.startswith(code):
                    return r.parent, r.asset_class, r.region
            return None, None, None

        # Build ID Map
        unique_syms = full_df[['instrument_id', 'raw_symbol']].drop_duplicates()
        id_map = {}
        for row in unique_syms.itertuples():
            parent, asset, region = get_root_info(row.raw_symbol)
            id_map[row.instrument_id] = (parent, asset, region)
            
        # Assign columns via mapping
        mapped_data = [id_map.get(i, (None, None, None)) for i in full_df['instrument_id']]
        
        full_df['parent'] = [x[0] for x in mapped_data]
        full_df['asset_class'] = [x[1] for x in mapped_data]
        full_df['region'] = [x[2] for x in mapped_data]
        
        # Final Cleanup: Sort and Reorder
        full_df = full_df.sort_values(['parent', 'date', 'expiration'])
        
        cols = ['date', 'parent', 'asset_class', 'region', 'symbol', 'expiration', 'open', 'high', 'low', 'close', 'volume', 'settlement', 'open_interest']
        remaining = [c for c in full_df.columns if c not in cols]
        full_df = full_df[cols + remaining]

        return full_df

if __name__ == "__main__":
    processor = DatasetProcessor()
    
    # Define paths to your data folders here
    # Example:
    # files = {
    #     "definition": DATA_DIR / "GLBX-20251203-K4BY8VWRT9",
    #     "ohlcv-1d": DATA_DIR / "GLBX-20251203-KWBSQ7EC9B",
    #     "statistics": DATA_DIR / "GLBX-20251203-FYW5M8HAT5"
    # }
    
    # logger.info("Please configure the 'files' dictionary in __main__ to point to your specific job folders.")
    # df_final = processor.process_data(files, GLBX_UNIVERSE)
    # df_final.to_csv(OUTPUT_FILE, index=False)
    # logger.info(f"Saved to {OUTPUT_FILE}")
    logger.info("Processor ready. Configure paths in __main__ to execute.")
