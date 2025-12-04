"""
Tests for the Dataset Processor.
"""
import pytest
import pandas as pd
from src.process_dataset import DatasetProcessor

def test_processor_init(tmp_path):
    """Test processor initialization."""
    processor = DatasetProcessor(data_dir=tmp_path)
    assert processor.data_dir == tmp_path

def test_load_empty_dir(tmp_path):
    """Test loading from an empty directory returns empty DataFrame."""
    processor = DatasetProcessor(data_dir=tmp_path)
    df = processor.load_data_from_dir(tmp_path)
    assert df.empty
