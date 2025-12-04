import os
import time
import logging
from pathlib import Path
import pandas as pd
import databento as db
from src.config import GLBX_UNIVERSE

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DOWNLOAD_DIR = Path("./databento_data")

def get_job_id(job):
    # Safely extract ID from either a Dictionary or a BatchJob object
    if isinstance(job, dict):
        return job.get('id')
    return job.id

def get_job_state(job):
    # Safely extract State from either a Dictionary or a BatchJob object
    if isinstance(job, dict):
        return job.get('state')
    return job.state

class FuturesDownloader:
    def __init__(self, api_key, data_dir=DOWNLOAD_DIR):
        self.client = db.Historical(api_key)
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def submit_and_download(self, universe, start_date, end_date):
        # Submits batch jobs for the universe, waits for completion, and downloads files
        dataset_roots = {}
        for r in universe:
            dataset_roots.setdefault(r.dataset, []).append(r.parent)

        job_map = {}
        
        for dataset, symbols in dataset_roots.items():
            logger.info(f"Submitting jobs for {dataset} ({len(symbols)} roots)...")
            
            def submit(schema):
                try:
                    job = self.client.batch.submit_job(
                        dataset=dataset,
                        symbols=symbols,
                        schema=schema,
                        start=start_date,
                        end=end_date,
                        stype_in="parent",
                        encoding="dbn",
                        compression="zstd"
                    )
                    return get_job_id(job)
                except Exception as e:
                    logger.error(f"Failed to submit {schema}: {e}")
                    raise RuntimeError(f"Failed to submit {schema}: {e}")

            jid_def = submit("definition")
            job_map["definition"] = jid_def
            logger.info(f"   -> Submitted Definition Job: {jid_def}")

            jid_ohlcv = submit("ohlcv-1d")
            job_map["ohlcv-1d"] = jid_ohlcv
            logger.info(f"   -> Submitted OHLCV Job: {jid_ohlcv}")

            jid_stats = submit("statistics")
            job_map["statistics"] = jid_stats
            logger.info(f"   -> Submitted Stats Job: {jid_stats}")

        logger.info(f"Waiting for {len(job_map)} jobs to complete...")
        
        file_paths = {}
        completed_jobs = set()
        
        while len(completed_jobs) < len(job_map):
            time.sleep(10)
            
            jobs_state = self.client.batch.list_jobs(since=pd.Timestamp.utcnow().floor('D'))
            
            for schema, jid in job_map.items():
                if jid in completed_jobs:
                    continue
                
                target_job = None
                for j in jobs_state:
                    if get_job_id(j) == jid:
                        target_job = j
                        break
                
                if not target_job:
                    continue
                
                state = get_job_state(target_job)
                
                if state == 'done':
                    logger.info(f"Downloading {schema} (Job {jid})...")
                    local_files = self.client.batch.download(
                        job_id=jid, 
                        output_dir=self.data_dir
                    )
                    file_paths[schema] = local_files[0]
                    completed_jobs.add(jid)
                
                elif state == 'failed':
                    logger.error(f"Job {jid} failed on Databento side.")
                    raise RuntimeError(f"Job {jid} failed on Databento side.")

        logger.info("All downloads complete.")
        return file_paths

if __name__ == "__main__":
    API_KEY = os.environ.get("DATABENTO_API_KEY")
    if not API_KEY:
        logger.error("DATABENTO_API_KEY env variable not set.")
        exit(1)

    downloader = FuturesDownloader(API_KEY)
    
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    
    logger.info(f"Starting download for {len(GLBX_UNIVERSE)} contracts...")
