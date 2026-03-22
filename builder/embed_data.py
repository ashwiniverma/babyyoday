#!/usr/bin/env python3
"""Chunk and embed a customer's documents into a FAISS index."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.reindex import reindex

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Embed customer documents into FAISS index"
    )
    parser.add_argument(
        "--docs-dir", required=True, help="Directory containing the customer's documents"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to write faiss.index, metadata.json, centroid.npy"
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    docs = Path(args.docs_dir)
    if not docs.exists():
        logger.error("Docs directory does not exist: %s", docs)
        sys.exit(1)

    file_count = sum(1 for f in docs.iterdir() if f.is_file())
    if file_count == 0:
        logger.error("No files found in %s", docs)
        sys.exit(1)

    logger.info("Processing %d files from %s", file_count, docs)
    reindex(
        docs_dir=str(docs),
        output_dir=args.output_dir,
        embedding_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info("Done. Index written to %s", args.output_dir)


if __name__ == "__main__":
    main()
