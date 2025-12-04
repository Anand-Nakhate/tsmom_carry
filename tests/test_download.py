"""
Tests for the Databento downloader.
"""
import pytest
from src.download import FuturesDownloader, get_job_id
from src.config import GLBX_UNIVERSE

def test_universe_config():
    """Verify universe is loaded correctly."""
    assert len(GLBX_UNIVERSE) > 0
    assert GLBX_UNIVERSE[0].dataset == "GLBX.MDP3"

def test_get_job_id_dict():
    """Test job ID extraction from dict."""
    job = {"id": "GLBX-123", "state": "done"}
    assert get_job_id(job) == "GLBX-123"

class MockJob:
    def __init__(self, jid):
        self.id = jid

def test_get_job_id_object():
    """Test job ID extraction from object."""
    job = MockJob("GLBX-456")
    assert get_job_id(job) == "GLBX-456"
