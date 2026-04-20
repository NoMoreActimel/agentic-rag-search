#!/usr/bin/env python3
"""Bootstrap and validate experiment data artifacts.

This script is intended for teammates who only need to run evaluation and
experiments without rebuilding the full data/indexing pipeline.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ZIP = PROJECT_ROOT.parent / "agentic-rag-search-data.zip"

REQUIRED_FILES = [
    "data/processed/qa_pairs.json",
    "data/processed/transcripts.parquet",
    "data/processed/transcripts_subset.parquet",
    "data/indices/chunks.parquet",
    "data/indices/bm25/params.index.json",
    "data/indices/embeddings/faiss.index",
    "data/indices/embeddings/embeddings.npy",
]


def _validate_data_layout(root: Path) -> list[str]:
    missing: list[str] = []
    for rel_path in REQUIRED_FILES:
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _validate_environment(root: Path) -> tuple[bool, str]:
    env_file = root / ".env"
    if not env_file.exists():
        return (
            False,
            ".env file missing. Copy from .env.example and add either GEMINI_API_KEY "
            "or Vertex settings (GOOGLE_GENAI_USE_VERTEXAI + GOOGLE_CLOUD_PROJECT).",
        )
    dotenv.load_dotenv(dotenv_path=env_file)
    use_vertex = _env_truthy(os.getenv("GOOGLE_GENAI_USE_VERTEXAI")) or _env_truthy(
        os.getenv("GEMINI_USE_VERTEX")
    )
    if use_vertex:
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            return (
                False,
                "Vertex AI enabled but GOOGLE_CLOUD_PROJECT not set in .env. "
                "See .env.example.",
            )
        return (
            True,
            "Vertex AI env vars present. Ensure: gcloud auth application-default login "
            "and Vertex AI API enabled for this project.",
        )
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        return False, "GEMINI_API_KEY (or GOOGLE_API_KEY) not found in .env."
    return True, "Environment looks good (Gemini Developer API / API key)."


def _extract_zip(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)


def _clear_existing_data(data_dir: Path) -> None:
    if data_dir.exists():
        shutil.rmtree(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and validate experiment data")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help="Path to data zip shared by teammate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing data directory before extracting zip.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate current workspace setup; do not extract anything.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip GEMINI_API_KEY validation (useful for offline data checks).",
    )
    args = parser.parse_args()

    if not args.check_only:
        if not args.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {args.zip_path}")
        if args.force:
            print("Removing existing data directory...")
            _clear_existing_data(DATA_DIR)
        print(f"Extracting {args.zip_path} into {PROJECT_ROOT} ...")
        _extract_zip(args.zip_path, PROJECT_ROOT)

    print("\nValidating data artifacts...")
    missing = _validate_data_layout(PROJECT_ROOT)
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All required data/index files are present.")

    if not args.skip_env_check:
        print("\nValidating environment...")
        ok, message = _validate_environment(PROJECT_ROOT)
        print(message)
        if not ok:
            raise SystemExit(1)
    else:
        print("\nSkipping environment validation as requested.")

    print("\nSetup validation complete.")


if __name__ == "__main__":
    main()
