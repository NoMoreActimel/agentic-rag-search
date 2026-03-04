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
cp .env.example .env  # Add your GEMINI_API_KEY
```

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
