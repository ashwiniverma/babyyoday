#!/usr/bin/env python3
"""
Build a customer's Docker image.

Usage:
    python builder/build_customer.py \
        --business-name "Sweet Rise Bakery" \
        --business-type bakery \
        --data ./sample_data/ \
        --model-path ./models/phi-3-mini.gguf \
        --tag sweetrise-agent:latest
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.reindex import reindex

logger = logging.getLogger(__name__)

BUILDER_DIR = Path(__file__).parent
PROJECT_ROOT = BUILDER_DIR.parent


def build_config(business_name: str, business_type: str) -> dict:
    """Generate a customer-specific config from the template."""
    with open(BUILDER_DIR / "config_template.yaml") as f:
        template = f.read()

    rendered = template.replace("{{ business_name }}", business_name)
    rendered = rendered.replace("{{ business_type }}", business_type)
    return yaml.safe_load(rendered)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build a customer's agent Docker image")
    parser.add_argument("--business-name", required=True, help="e.g. 'Sweet Rise Bakery'")
    parser.add_argument("--business-type", default="general", help="e.g. bakery, law, saas")
    parser.add_argument("--data", required=True, help="Directory with customer's documents")
    parser.add_argument("--model-path", default=None, help="Path to .gguf model file (optional)")
    parser.add_argument("--tag", default="babyyoday-agent:latest", help="Docker image tag")
    parser.add_argument("--output-dir", default=None, help="Staging dir (default: /tmp/babyyoday_build)")
    args = parser.parse_args()

    staging = Path(args.output_dir or "/tmp/babyyoday_build")
    staging.mkdir(parents=True, exist_ok=True)

    data_staging = staging / "data"
    docs_staging = data_staging / "docs"
    incoming_staging = data_staging / "incoming"
    models_staging = staging / "models"

    for d in [docs_staging, incoming_staging, models_staging]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy customer documents
    logger.info("Copying documents from %s ...", args.data)
    src = Path(args.data)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, docs_staging / f.name)

    # 2. Build FAISS index + centroid
    logger.info("Building index ...")
    reindex(
        docs_dir=str(docs_staging),
        output_dir=str(data_staging),
    )

    # 3. Copy model if provided
    if args.model_path:
        model_file = Path(args.model_path)
        if model_file.exists():
            shutil.copy2(model_file, models_staging / "model.gguf")
            logger.info("Model copied: %s", model_file.name)
        else:
            logger.warning("Model file not found: %s — container will run in retrieval-only mode", args.model_path)

    # 4. Generate config
    config = build_config(args.business_name, args.business_type)
    config_path = staging / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Config written to %s", config_path)

    # 5. Copy application code
    app_staging = staging / "app_code"
    for module in ["inference", "agent", "admin", "data_pipeline"]:
        src_module = PROJECT_ROOT / module
        dst_module = app_staging / module
        if src_module.exists():
            shutil.copytree(src_module, dst_module, dirs_exist_ok=True)

    shutil.copy2(
        PROJECT_ROOT / "inference" / "requirements.txt",
        staging / "requirements.txt",
    )

    # 6. Copy Dockerfile
    shutil.copy2(BUILDER_DIR / "Dockerfile", staging / "Dockerfile")

    # 7. Build Docker image
    logger.info("Building Docker image: %s ...", args.tag)
    result = subprocess.run(
        ["docker", "build", "-t", args.tag, "."],
        cwd=str(staging),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Image built successfully: %s", args.tag)
    logger.info(
        "Run with: docker run -p 8000:8000 -p 8001:8001 %s", args.tag
    )


if __name__ == "__main__":
    main()
