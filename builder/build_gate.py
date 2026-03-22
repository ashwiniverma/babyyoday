#!/usr/bin/env python3
"""Compute the domain centroid from an existing FAISS index + metadata."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build domain gate centroid from existing index"
    )
    parser.add_argument("--index-path", required=True, help="Path to faiss.index")
    parser.add_argument("--metadata-path", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Path to write centroid.npy")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    with open(args.metadata_path) as f:
        metadata = json.load(f)

    logger.info("Re-embedding %d chunks for centroid computation ...", len(metadata))
    model = SentenceTransformer(args.model)
    texts = [m["text"] for m in metadata]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    np.save(args.output, centroid)
    logger.info("Centroid saved to %s (dim=%d)", args.output, centroid.shape[0])


if __name__ == "__main__":
    main()
