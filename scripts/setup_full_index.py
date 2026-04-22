#!/usr/bin/env python3
"""One-shot teammate setup: extract the 426-episode data zip into this repo.

After `git clone` + `.env` (GEMINI_API_KEY or Vertex creds), run:

    python scripts/setup_full_index.py --zip path/to/agentic-rag-search-data-426ep.zip

Replaces the index (chunks, BM25, FAISS, embeddings), the 87-pair era-tagged
qa_pairs.json, all 69 episode metadata files, and transcripts.parquet. Quality
chunk scores come from git (data/processed/chunk_quality_scores.json is tracked).

You can then run scripts/05_run_experiments.py as usual.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = PROJECT_ROOT / "agentic-rag-search-data-426ep.zip"

# Files/dirs replaced on extract. Everything else in data/ is left alone.
STALE_PATHS = [
    "data/indices/chunks.parquet",
    "data/indices/bm25",
    "data/indices/embeddings/embeddings.npy",
    "data/indices/embeddings/faiss.index",
    "data/processed/qa_pairs.json",
    "data/processed/metadata",
    "data/processed/transcripts.parquet",
]

REQUIRED_AFTER = [
    "data/indices/chunks.parquet",
    "data/indices/bm25/params.index.json",
    "data/indices/embeddings/faiss.index",
    "data/indices/embeddings/embeddings.npy",
    "data/processed/qa_pairs.json",
    "data/processed/transcripts.parquet",
    # Quality scores come from git, not the zip
    "data/processed/chunk_quality_scores.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                   help=f"Path to the data zip (default: {DEFAULT_ZIP.name} in repo root).")
    p.add_argument("--yes", action="store_true",
                   help="Skip the overwrite confirmation prompt.")
    return p.parse_args()


def clear_stale(project_root: Path) -> None:
    for rel in STALE_PATHS:
        p = project_root / rel
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  removed dir  {rel}/")
        elif p.exists():
            p.unlink()
            print(f"  removed file {rel}")


def extract(zip_path: Path, project_root: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(project_root)
        return len(zf.namelist())


def validate(project_root: Path) -> list[str]:
    problems = []
    for rel in REQUIRED_AFTER:
        if not (project_root / rel).exists():
            problems.append(f"missing: {rel}")
    if problems:
        return problems

    chunks = pd.read_parquet(project_root / "data/indices/chunks.parquet")
    emb = np.load(project_root / "data/indices/embeddings/embeddings.npy", mmap_mode="r")
    if emb.shape[0] != len(chunks):
        problems.append(f"alignment mismatch: embeddings.npy has {emb.shape[0]} rows, "
                        f"chunks.parquet has {len(chunks)}")

    with open(project_root / "data/processed/qa_pairs.json") as f:
        qa = json.load(f)
    if not qa or "era" not in qa[0]:
        problems.append("qa_pairs.json missing era tags — not the new dataset")

    return problems


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        print(f"ERROR: zip not found: {args.zip}")
        sys.exit(1)

    print(f"Applying zip: {args.zip}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("This will REPLACE:")
    for rel in STALE_PATHS:
        print(f"  - {rel}")
    print()

    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    print("\n1. Clearing stale paths ...")
    clear_stale(PROJECT_ROOT)

    print(f"\n2. Extracting {args.zip.name} ...")
    n = extract(args.zip, PROJECT_ROOT)
    print(f"   {n} entries extracted")

    print("\n3. Validating ...")
    problems = validate(PROJECT_ROOT)
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)

    chunks = pd.read_parquet(PROJECT_ROOT / "data/indices/chunks.parquet")
    with open(PROJECT_ROOT / "data/processed/qa_pairs.json") as f:
        qa = json.load(f)
    from collections import Counter
    era_counts = Counter(p.get("era", "?") for p in qa)

    print("   OK")
    print(f"   chunks.parquet: {len(chunks):,} rows across {chunks['episode_id'].nunique()} episodes")
    print(f"   qa_pairs.json:  {len(qa)} pairs  (era counts: {dict(era_counts)})")
    print()
    print("Ready. Next:")
    print("  python scripts/05_run_experiments.py --help")


if __name__ == "__main__":
    main()
