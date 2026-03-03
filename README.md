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
python scripts/01_build_raw_dataset.py   # Build raw transcripts
python scripts/02_extract_metadata.py    # Extract metadata with Gemini
python scripts/03_generate_qa.py         # Generate Q&A pairs
python scripts/04_create_indices.py      # Create BM25 + embedding indices
```
