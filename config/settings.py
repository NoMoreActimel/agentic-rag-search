"""Central configuration for all paths, model names, and parameters."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_HF_DIR = RAW_DIR / "huggingface"
RAW_SCRAPED_DIR = RAW_DIR / "scraped"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = PROCESSED_DIR / "metadata"
INDICES_DIR = DATA_DIR / "indices"

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
        METADATA_DIR, INDICES_DIR, BM25_DIR, EMBEDDINGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
