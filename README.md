# Process-Level Feedback for Agentic Information Search

## Problem

Current RAG evaluations focus on final retrieval quality but ignore the *search process* — how an agentic system iteratively queries, evaluates, and refines its retrieval strategy. This project builds infrastructure for evaluating agentic RAG over podcast transcripts, where questions require multi-step reasoning across episodes.

## Methodology

1. **Dataset Construction** — Merge transcripts from HuggingFace (episodes 1-325) and web-scraped data (episodes 326+) from the Lex Fridman Podcast into a unified corpus.
2. **Metadata Extraction** — Use Gemini to extract structured metadata (entities, topics, summaries) per episode.
3. **Synthetic Q&A Generation** — Generate 4 types of questions that require different retrieval strategies:
  - **Type 1: Multi-Hop Bridge** — Questions requiring entity-linking across episodes (e.g., "What did the inventor of X say about Y?")
  - **Type 2: Comparative Viewpoint** — Questions comparing perspectives of different guests on shared topics
  - **Type 3: Temporal Evolution** — Questions tracking how a guest's views changed across appearances
  - **Type 4: Quantitative Aggregation** — Questions requiring exhaustive retrieval of numeric claims
4. **Search Index Creation** — Build BM25 and embedding-based indices over chunked transcripts for hybrid retrieval.

## Dataset

- ~450+ episodes from the Lex Fridman Podcast
- Working subset: 50 episodes (25 from HuggingFace, 25 from web scraping)
- 20 Q&A pairs for initial iteration (5 per type)

## Setup

```bash
conda env create -f environment.yml
conda activate agentic-rag
cp .env.example .env
```

Then choose **one** auth mode in `.env` (see `.env.example`):

- **Gemini Developer API (Google AI Studio):** set `GEMINI_API_KEY` — billed on AI Studio / API key tier.
- **Gemini on Vertex AI (GCP / free trial credits):** set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`, then use **Application Default Credentials** (no AI Studio API key required for requests). Official overview: [Google Gen AI SDK — Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview).

### Gemini on Vertex AI (quick checklist)

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and log in: `gcloud auth application-default login`
2. Pick your **GCP project ID** (Console home → project selector; numeric *project number* is not the same as **project ID**).
3. Enable **Vertex AI API** for that project: Cloud Console → APIs & Services → Enable **Vertex AI API**.
4. In `.env` set for example:
   - `GOOGLE_GENAI_USE_VERTEXAI=true`
   - `GOOGLE_CLOUD_PROJECT=your-project-id`
   - `GOOGLE_CLOUD_LOCATION=us-central1`
5. (Optional) Service account JSON instead of ADC: set `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json` and grant the account **Vertex AI User** (or broader) on the project.

The code still uses the **`google-genai`** package; only client initialization switches to Vertex when the env vars above are set. Agent, judge, and evaluator call sites are unchanged.

**Note:** GCP trial credits apply to **Google Cloud** (e.g. Vertex), not to the separate **Google AI Studio** paid tier tied to `GEMINI_API_KEY`.

## Pipeline

```bash
# Step 1: Build raw transcripts (HuggingFace + web scraping)
python scripts/01_build_raw_dataset.py

# Step 2: Extract metadata with Gemini (requires GEMINI_API_KEY)
python scripts/02_extract_metadata.py

# Step 3a: Find Q&A candidates (pure Python, no API calls)
python scripts/03a_find_qa_candidates.py

# Step 3b: Generate Q&A pairs from candidates (requires GEMINI_API_KEY)
python scripts/03b_generate_qa.py

# Step 4: Create BM25 + embedding search indices (requires GEMINI_API_KEY)
python scripts/04_create_indices.py

# Step 4b (optional): Precompute chunk quality scores for quality_reweight experiments
python scripts/04b_score_chunks.py

# Step 5: Run evaluation experiments (smoke run)
python scripts/05_run_experiments.py --run-main-grid --mode smoke --num-questions 3 --max-steps-values 2,3,4
```

## For Colleagues: Creating Search Indices

If the raw dataset, metadata, and Q&A pairs are already generated (steps 1-3), you only need to run step 4 to create the search indices:

```bash
conda env create -f environment.yml
conda activate agentic-rag
cp .env.example .env  # Add your GEMINI_API_KEY
python scripts/04_create_indices.py
```

This will:

1. Chunk the 50-episode subset into ~1000-char segments with 150-char overlap
2. Build a BM25 index (saved to `data/indices/bm25/`)
3. Generate Gemini embeddings (768 dims) and build a FAISS index (saved to `data/indices/embeddings/`)

The embedding step calls the Gemini API and takes ~5 minutes. It has checkpointing built in, so if it's interrupted by rate limits it will resume from the last checkpoint.

## Evaluation Branch (Branch 3)

### 1) Bootstrap data from teammate archive

If you received `agentic-rag-search-data.zip`, bootstrap and validate in one step:

```bash
python scripts/00_setup_experiment_data.py --zip-path ../agentic-rag-search-data.zip
```

Validate only (no extraction):

```bash
python scripts/00_setup_experiment_data.py --check-only
```

Validate data only (skip API key check):

```bash
python scripts/00_setup_experiment_data.py --check-only --skip-env-check
```

### 2) Run strict 36-condition main grid

The main grid is:

- 3 retrievers: `grep`, `bm25`, `embedding`
- 3 iteration settings: `max_steps in {2,3,4}`
- process feedback: on/off
- quality reweight: on/off

```bash
python scripts/05_run_experiments.py \
  --run-main-grid \
  --mode full \
  --max-steps-values 2,3,4 \
  --top-k 5
```

Run only the quality-off arm (18 conditions) while chunk quality scoring is still in progress:

```bash
python scripts/05_run_experiments.py \
  --run-main-grid \
  --only-quality-off \
  --mode smoke \
  --num-questions 3 \
  --max-steps-values 2,3,4 \
  --top-k 5
```

### 3) Run oracle mini-study (separate phase)

```bash
python scripts/05_run_experiments.py \
  --run-oracle-mini-study \
  --mode smoke \
  --num-questions 3
```

### 4) Run quick smoke for pipeline verification

```bash
python scripts/05_run_experiments.py \
  --run-main-grid \
  --run-oracle-mini-study \
  --mode smoke \
  --num-questions 2 \
  --max-steps-values 2,3,4 \
  --output-tag smoke_check
```

### 5) Outputs and report artifacts

Each run creates `data/results/<timestamp>[_tag]/` with:

- `manifest.json`: run configuration and dataset references
- `conditions_main.json` / `conditions_oracle_mini.json`: executed condition definitions
- `runs.jsonl`: per-example records (config, answers, metrics, usage deltas)
- `trajectories/*.json`: full trajectory per example-condition pair
- `per_example_metrics.csv`: flattened table for analysis
- `summary_by_condition.csv`: aggregate metrics by condition
- `summary_overview.json`: top-level KPI snapshot
- `gemini_usage_final.json`: cumulative API usage/cost counters

### 6) Basic verification checks

```bash
python -m unittest discover -s tests -p "test_*.py"
```

