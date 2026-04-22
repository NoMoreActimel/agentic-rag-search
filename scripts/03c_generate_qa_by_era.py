#!/usr/bin/env python3
"""Generate QA pairs stratified by cohort-era, using all metadata for candidates.

Splits episode metadata by the Gemini 2.5 Flash knowledge cutoff (2025-01-31):
- "old"   : all reference episodes are pre-cutoff (or undated HF archive)
- "new"   : all reference episodes are post-cutoff (unseen by the model)
- "mixed" : references span both eras

Each QA pair is stamped with `era`. For every QA type we target a fixed count
per era (default 8 each → 96 pairs total at 8×3×4). Candidates from the
generators' `find_candidates()` are partitioned by era and drawn down per bucket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import QA_PAIRS_JSON, TRANSCRIPTS_PARQUET
from src.llm.gemini_client import GeminiClient
from src.qa.qa_generator import QAPair, QATypeGenerator, load_all_metadata
from src.qa.type1_multihop import MultiHopGenerator
from src.qa.type2_comparative import ComparativeGenerator
from src.qa.type3_temporal import TemporalGenerator
from src.qa.type4_aggregation import AggregationGenerator

CUTOFF = pd.Timestamp("2025-01-31", tz="UTC")
ERAS = ("old", "new", "mixed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-era-per-type", type=int, default=8,
                   help="Target pairs for each (era, QA type) bucket.")
    p.add_argument("--oversample", type=int, default=12,
                   help="Candidate oversampling multiplier passed to find_candidates.")
    p.add_argument("--output", type=Path, default=QA_PAIRS_JSON)
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--preview-only", action="store_true",
                   help="Generate 2 per (era, type) instead of the full count for sanity-check.")
    return p.parse_args()


def build_post_cutoff_set() -> set[int]:
    df = pd.read_parquet(TRANSCRIPTS_PARQUET)
    df["dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return set(int(e) for e in df[df["dt"] > CUTOFF]["episode_id"].tolist())


def classify_era(episode_ids: list[int], post_cutoff: set[int]) -> str:
    eps = set(int(e) for e in episode_ids)
    in_new = eps & post_cutoff
    in_old = eps - post_cutoff
    if in_new and not in_old:
        return "new"
    if in_old and not in_new:
        return "old"
    return "mixed"


async def generate_bucketed_for_type(
    generator: QATypeGenerator,
    per_era_count: int,
    oversample: int,
    post_cutoff: set[int],
) -> list[QAPair]:
    """Fill {old, new, mixed} buckets for one QA type from a single candidate pool."""
    # Oversample heavily — we want enough candidates in each era bucket.
    target_total = per_era_count * 3
    candidates = generator.find_candidates(count=target_total * oversample)
    buckets: dict[str, list] = defaultdict(list)
    for c in candidates:
        buckets[classify_era(c.episode_ids, post_cutoff)].append(c)

    print(f"  [{generator.qa_type}] candidates by era: "
          f"old={len(buckets['old'])}  new={len(buckets['new'])}  mixed={len(buckets['mixed'])}")

    pairs_by_era: dict[str, list[QAPair]] = {e: [] for e in ERAS}
    for era in ERAS:
        for candidate in buckets.get(era, []):
            if len(pairs_by_era[era]) >= per_era_count:
                break
            pair = await generator.generate_from_candidate(candidate)
            if pair is None:
                continue
            # Re-classify from reference_episodes (generator may deviate from candidate).
            actual_era = classify_era(pair.reference_episodes, post_cutoff)
            pair.era = actual_era
            pairs_by_era[actual_era].append(pair)

        got = len(pairs_by_era[era])
        if got < per_era_count:
            print(f"  [{generator.qa_type}] era={era}: only {got}/{per_era_count} "
                  f"(candidate pool exhausted)")

    return [p for era in ERAS for p in pairs_by_era[era][:per_era_count]]


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f".bak.{stamp}.json")
    shutil.copy2(path, backup)
    return backup


async def main() -> None:
    args = parse_args()
    per_era = 2 if args.preview_only else args.per_era_per_type

    metadata = load_all_metadata()
    if not metadata:
        print("No metadata in data/processed/metadata/ — run 02_extract_metadata first.")
        sys.exit(1)

    post_cutoff = build_post_cutoff_set()
    ep_era_counts = Counter(
        classify_era([m["episode_id"]], post_cutoff) for m in metadata
    )
    print(f"Loaded metadata for {len(metadata)} episodes  "
          f"(old={ep_era_counts['old']}  new={ep_era_counts['new']})")

    if not args.no_backup:
        bk = backup_existing(args.output)
        if bk:
            print(f"Backed up previous QA pairs to {bk}")

    client = GeminiClient()
    generators: list[QATypeGenerator] = [
        MultiHopGenerator(client, metadata),
        ComparativeGenerator(client, metadata),
        TemporalGenerator(client, metadata),
        AggregationGenerator(client, metadata),
    ]

    all_pairs: list[QAPair] = []
    for gen in generators:
        print(f"\n=== {gen.qa_type} — target {per_era} per era ===")
        pairs = await generate_bucketed_for_type(
            generator=gen,
            per_era_count=per_era,
            oversample=args.oversample,
            post_cutoff=post_cutoff,
        )
        all_pairs.extend(pairs)
        by_era = Counter(p.era for p in pairs)
        print(f"  produced {len(pairs)}: {dict(by_era)}")

    # Save
    output = [p.to_dict() for p in all_pairs]
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_pairs)} QA pairs to {args.output}")

    print("\n=== Final breakdown ===")
    table = Counter((p.qa_type, p.era) for p in all_pairs)
    for t in ("multihop", "comparative", "temporal", "aggregation"):
        row = {era: table.get((t, era), 0) for era in ERAS}
        print(f"  {t:<12}  old={row['old']:>2}  new={row['new']:>2}  mixed={row['mixed']:>2}")

    client.print_stats()


if __name__ == "__main__":
    asyncio.run(main())
