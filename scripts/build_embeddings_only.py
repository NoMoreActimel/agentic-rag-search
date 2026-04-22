#!/usr/bin/env python3
"""Resume: build only the embedding index over existing chunks.parquet (66K rows)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CHUNKS_PARQUET, EMBEDDINGS_DIR
from src.indexing.embedding_index import build_embedding_index


async def main() -> None:
    idx = await build_embedding_index(
        chunks_path=str(CHUNKS_PARQUET),
        output_dir=str(EMBEDDINGS_DIR),
    )
    print(f"FAISS vectors built: {idx.ntotal}")


if __name__ == "__main__":
    asyncio.run(main())
