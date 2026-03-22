#!/usr/bin/env python3
"""
Quick local setup: index the sample data so you can run the server locally.

Usage:
    pip install -r requirements.txt
    python setup_local.py
    uvicorn inference.server:app --port 8000
"""
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_data"
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
INCOMING_DIR = DATA_DIR / "incoming"


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Copying sample data to %s ...", DOCS_DIR)
    for f in SAMPLE_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, DOCS_DIR / f.name)

    logger.info("Building FAISS index ...")
    from data_pipeline.reindex import reindex

    reindex(
        docs_dir=str(DOCS_DIR),
        output_dir=str(DATA_DIR),
    )

    logger.info("Local setup complete!")
    logger.info("")
    logger.info("Start the server with:")
    logger.info("  uvicorn inference.server:app --port 8000 --reload")
    logger.info("")
    logger.info("Start the admin panel with:")
    logger.info("  uvicorn admin.app:admin_app --port 8001 --reload")
    logger.info("")
    logger.info("Test a query:")
    logger.info('  curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d \'{"query": "Do you have vegan options?"}\'')


if __name__ == "__main__":
    main()
