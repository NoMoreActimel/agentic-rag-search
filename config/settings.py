"""Central configuration for all paths, model names, and parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


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
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_DIMS = 768
GEMINI_RPM_LIMIT = 140
GEMINI_CONCURRENT_LIMIT = 5
GEMINI_MAX_RETRIES = 3

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
