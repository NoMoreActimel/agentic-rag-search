"""Central configuration for all paths, model names, and parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    """Parse a positive integer from the environment with bounds."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return default
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_HF_DIR = RAW_DIR / "huggingface"
RAW_SCRAPED_DIR = RAW_DIR / "scraped"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = PROCESSED_DIR / "metadata"
INDICES_DIR = DATA_DIR / "indices"
RESULTS_DIR = DATA_DIR / "results"

TRANSCRIPTS_PARQUET = PROCESSED_DIR / "transcripts.parquet"
TRANSCRIPTS_SUBSET_PARQUET = PROCESSED_DIR / "transcripts_subset.parquet"
QA_PAIRS_JSON = PROCESSED_DIR / "qa_pairs.json"
CHUNKS_PARQUET = INDICES_DIR / "chunks.parquet"
BM25_DIR = INDICES_DIR / "bm25"
EMBEDDINGS_DIR = INDICES_DIR / "embeddings"

# ── HuggingFace dataset ──────────────────────────────────────────────────────
HF_DATASET_NAME = "nmac/lex_fridman_podcast"

# ── Web scraper ───────────────────────────────────────────────────────────────
LEX_PODCAST_URL = "https://lexfridman.com/podcast/"
SCRAPE_DELAY_SECONDS = 2.0
SCRAPE_MAX_RETRIES = 3

# ── Gemini LLM ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_QUALITY_MODEL = "gemini-2.5-flash-lite"  # Cheaper model for offline chunk quality scoring; same price as 2.0 Flash but proper Tier 1 throughput
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_DIMS = 768
# Throughput / stability (override via .env — see .env.example)
GEMINI_RPM_LIMIT = _env_int("GEMINI_RPM_LIMIT", 140, 1, 100_000)
GEMINI_CONCURRENT_LIMIT = _env_int("GEMINI_CONCURRENT_LIMIT", 6, 1, 128)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 4, 1, 12)
# Token-bucket burst (requests) for average RPM; larger = burstier starts, still capped by sustained RPM.
_default_rpm_burst = max(8, min(28, GEMINI_CONCURRENT_LIMIT * 2))
GEMINI_RPM_BURST_CAPACITY = _env_int("GEMINI_RPM_BURST_CAPACITY", _default_rpm_burst, 1, 500)
# scripts/05_run_experiments.py defaults when --qa-concurrency / --eval-concurrency omitted
GEMINI_QA_CONCURRENCY_DEFAULT = _env_int("GEMINI_QA_CONCURRENCY_DEFAULT", 4, 1, 64)
GEMINI_EVAL_CONCURRENCY = _env_int("GEMINI_EVAL_CONCURRENCY", 3, 1, 32)
GEMINI_EMBED_MAX_RETRIES = _env_int("GEMINI_EMBED_MAX_RETRIES", 5, 1, 12)
# Higher limits for offline chunk quality scoring on 2.5-Flash-Lite Tier 1
# (AI Studio). Empirical throughput ~50 chunks/s at concurrent=80 with the
# native-async client path; zero 429s observed at these settings.
GEMINI_QUALITY_RPM_LIMIT = _env_int("GEMINI_QUALITY_RPM_LIMIT", 3000, 1, 100_000)
GEMINI_QUALITY_CONCURRENT_LIMIT = _env_int("GEMINI_QUALITY_CONCURRENT_LIMIT", 80, 1, 256)
# Per-request HTTP timeout for google-genai (milliseconds). Prevents hung calls from blocking the full grid.
GEMINI_HTTP_TIMEOUT_MS = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "600000"))  # default 10 minutes

# ── Vertex AI (Gemini API on Vertex) vs Gemini Developer API (API key) ───────
# Official env pattern: https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
# Set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION,
# then use Application Default Credentials (gcloud auth application-default login).
GEMINI_USE_VERTEX_AI: bool = _env_truthy(os.getenv("GOOGLE_GENAI_USE_VERTEXAI")) or _env_truthy(
    os.getenv("GEMINI_USE_VERTEX")
)
GOOGLE_CLOUD_PROJECT: str | None = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 150     # characters

# ── Embedding batches ─────────────────────────────────────────────────────────
EMBEDDING_BATCH_SIZE = 50

# ── Subset selection ──────────────────────────────────────────────────────────
SUBSET_SIZE = 50
SUBSET_HF_COUNT = 25
SUBSET_SCRAPED_COUNT = 25

# ── Q&A generation ────────────────────────────────────────────────────────────
QA_PAIRS_PER_TYPE = 5
QA_TYPES = ["multihop", "comparative", "temporal", "aggregation"]

# ── Ensure directories exist ─────────────────────────────────────────────────
def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        RAW_HF_DIR, RAW_SCRAPED_DIR, PROCESSED_DIR,
        METADATA_DIR, INDICES_DIR, BM25_DIR, EMBEDDINGS_DIR, RESULTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
