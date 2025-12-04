"""
Tests for the Data Cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
from src.cleaning import load_and_clean, run_sanity_checks, generate_universe_audit

@pytest.fixture
def sample_df():
    """Creates a sample DataFrame for testing."""
    data = {
        'parent': ['ES.FUT', 'ES.FUT', 'ES.FUT'],
        'symbol': ['ESZ5', 'ESZ5', 'ESZ5'],
        'date': ['2025-12-01', '2025-12-02', '2025-12-01'], # Duplicate date
        'open': [100.0, 101.0, 100.0],
        'high': [105.0, 106.0, 105.0],
        'low': [95.0, 96.0, 95.0],
        'close': [102.0, 103.0, 102.0],
        'volume': [1000, 1500, 1000],
        'instrument_id': [1, 1, 1]
    }
    return pd.DataFrame(data)

def test_deduplication(tmp_path, sample_df):
    """Test that exact duplicates are removed."""
    # Save sample to CSV
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    
    # Load and clean
    cleaned_df = load_and_clean(csv_path)
    
    # Should have 2 rows (one duplicate removed)
    assert len(cleaned_df) == 2
    assert cleaned_df.iloc[0]['date'] == pd.Timestamp('2025-12-01')

def test_sanity_checks_bad_ohlc():
    """Test detection of inconsistent OHLC."""
    df = pd.DataFrame({
        'open': [100], 'high': [90], 'low': [80], 'close': [95] # High < Open
    })
    issues = run_sanity_checks(df)
    assert issues['bad_ohlc_rows'] == 1

def test_sanity_checks_neg_price():
    """Test detection of negative prices."""
    df = pd.DataFrame({
        'close': [-10.0, 50.0],
        'parent': ['CL.FUT', 'ES.FUT']
    })
    issues = run_sanity_checks(df)
    assert issues['neg_price_rows'] == 1
    assert issues['neg_price_roots']['CL.FUT'] == 1

def test_universe_audit():
    """Test audit generation."""
    df = pd.DataFrame({
        'parent': ['A', 'A', 'B'],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01']),
        'volume': [100, 200, 300]
    })
    audit = generate_universe_audit(df)
    
    assert len(audit) == 2
    assert audit.loc['A', 'n_days'] == 2
    assert audit.loc['B', 'avg_volume'] == 300.0
