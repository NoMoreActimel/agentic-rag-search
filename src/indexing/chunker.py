"""Sentence-boundary-aware chunking of transcripts."""

import re

import pandas as pd
from tqdm import tqdm

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PARQUET,
    TRANSCRIPTS_SUBSET_PARQUET,
)

# Sentence boundary pattern: period/question/exclamation followed by space and uppercase
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence boundaries."""
    sentences = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _hard_split_long_sentence(sentence: str, max_len: int) -> list[str]:
    """Split a sentence that exceeds max_len at word boundaries (fallback to char-level)."""
    if len(sentence) <= max_len:
        return [sentence]
    pieces: list[str] = []
    words = sentence.split(" ")
    buf: list[str] = []
    buf_len = 0
    for w in words:
        add_len = len(w) + (1 if buf else 0)
        if buf and buf_len + add_len > max_len:
            pieces.append(" ".join(buf))
            buf, buf_len = [w], len(w)
        else:
            buf.append(w)
            buf_len += add_len
    if buf:
        pieces.append(" ".join(buf))
    # If a single word was longer than max_len, char-split the oversized pieces.
    final: list[str] = []
    for p in pieces:
        if len(p) <= max_len:
            final.append(p)
        else:
            for i in range(0, len(p), max_len):
                final.append(p[i : i + max_len])
    return final


def chunk_transcript(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split a transcript into overlapping chunks at sentence boundaries.

    Each chunk is approximately `chunk_size` characters, with `overlap`
    characters of overlap between consecutive chunks. Splits happen
    at sentence boundaries to preserve readability.
    """
    raw_sentences = _split_into_sentences(text)
    if not raw_sentences:
        return [text[:chunk_size]] if text.strip() else []

    # Guarantee no single sentence exceeds chunk_size; this defends against transcripts
    # that lack ". Uppercase" boundaries (e.g. all-lowercase or missing punctuation) —
    # otherwise one pathological input produces a chunk many times larger than chunk_size.
    sentences: list[str] = []
    for s in raw_sentences:
        sentences.extend(_hard_split_long_sentence(s, chunk_size))

    chunks = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence exceeds chunk size, finalize current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Create overlap by keeping trailing sentences
            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s) + 1  # +1 for space

            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_chunks(
    subset_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Chunk all transcripts in the subset and save to parquet.

    Each chunk gets metadata: chunk_id, episode_id, guest, date, chunk_index, text.
    """
    subset_path = subset_path or str(TRANSCRIPTS_SUBSET_PARQUET)
    output_path = output_path or str(CHUNKS_PARQUET)

    df = pd.read_parquet(subset_path)
    all_chunks = []
    chunk_id = 0

    for _, row in tqdm(list(df.iterrows()), desc="Chunking transcripts"):
        transcript = str(row["full_transcript"])
        chunks = chunk_transcript(transcript)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "episode_id": int(row["episode_id"]),
                "guest": str(row.get("guest", "")),
                "date": row.get("date"),
                "chunk_index": idx,
                "text": chunk_text,
            })
            chunk_id += 1

    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_parquet(output_path, index=False)

    print(f"Created {len(chunks_df)} chunks from {len(df)} episodes")
    print(f"Avg chunk length: {chunks_df['text'].str.len().mean():.0f} chars")
    print(f"Saved to {output_path}")

    return chunks_df
