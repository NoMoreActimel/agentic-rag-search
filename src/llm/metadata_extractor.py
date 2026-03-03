"""Extract structured metadata from podcast transcripts using Gemini."""

import asyncio
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.settings import METADATA_DIR, TRANSCRIPTS_SUBSET_PARQUET
from src.llm.gemini_client import GeminiClient

SYSTEM_INSTRUCTION = """You are a metadata extraction system for podcast transcripts.
Given a full podcast transcript, extract structured information in JSON format.
Be thorough but precise — only include information explicitly stated or strongly implied in the transcript."""

EXTRACTION_PROMPT = """Analyze the following podcast transcript and extract structured metadata.

Episode: #{episode_id} — "{title}" with {guest}

Transcript:
{transcript}

---

Extract the following as JSON:
{{
  "episode_id": {episode_id},
  "guest_info": {{
    "name": "Full name of the guest",
    "title": "Professional title or affiliation mentioned",
    "expertise": ["List of areas of expertise discussed"]
  }},
  "summary": "A 2-3 sentence summary of the episode's main discussion",
  "main_entities": [
    {{
      "name": "Entity name",
      "type": "person|project|concept|organization|technology",
      "context": "Brief description of how this entity was discussed"
    }}
  ],
  "key_details": [
    {{
      "fact": "A specific factual claim or detail mentioned",
      "topic": "The topic area this relates to"
    }}
  ],
  "topics": ["List of main topics discussed"],
  "other_persons_mentioned": [
    {{
      "name": "Name of person mentioned",
      "context": "Why they were mentioned"
    }}
  ]
}}

Include 5-15 main entities, 5-10 key details, and all significant persons mentioned.
Return ONLY valid JSON."""


async def extract_episode_metadata(
    client: GeminiClient,
    episode_id: int,
    title: str,
    guest: str,
    transcript: str,
) -> dict | None:
    """Extract metadata for a single episode."""
    # Truncate very long transcripts to stay within context limits
    max_chars = 200_000  # ~50K tokens, well within Gemini's 1M context
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated]"

    prompt = EXTRACTION_PROMPT.format(
        episode_id=episode_id,
        title=title,
        guest=guest,
        transcript=transcript,
    )

    try:
        response = await client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        )
        metadata = json.loads(response)
        metadata["episode_id"] = episode_id  # Ensure correct ID
        return metadata
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Error extracting metadata for episode {episode_id}: {e}")
        return None


async def extract_all_metadata(
    subset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Extract metadata for all episodes in the subset.

    Implements checkpointing: skips episodes with existing metadata files.
    """
    subset_path = Path(subset_path or TRANSCRIPTS_SUBSET_PARQUET)
    output_dir = Path(output_dir or METADATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(subset_path)
    client = GeminiClient()

    results = []
    skipped = 0

    for _, row in tqdm(list(df.iterrows()), desc="Extracting metadata"):
        ep_id = int(row["episode_id"])
        output_file = output_dir / f"{ep_id}.json"

        # Checkpoint: skip if already processed
        if output_file.exists():
            skipped += 1
            with open(output_file) as f:
                results.append(json.load(f))
            continue

        metadata = await extract_episode_metadata(
            client=client,
            episode_id=ep_id,
            title=str(row["title"]),
            guest=str(row["guest"]),
            transcript=str(row["full_transcript"]),
        )

        if metadata:
            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=2)
            results.append(metadata)

    if skipped:
        print(f"Skipped {skipped} episodes (already processed)")

    client.print_stats()
    return results
