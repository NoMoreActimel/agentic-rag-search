#!/usr/bin/env python3
"""Run evaluation experiments for agentic RAG configurations."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    GEMINI_CONCURRENT_LIMIT,
    GEMINI_EVAL_CONCURRENCY,
    GEMINI_QA_CONCURRENCY_DEFAULT,
    GEMINI_RPM_BURST_CAPACITY,
    GEMINI_RPM_LIMIT,
    QA_PAIRS_JSON,
    RESULTS_DIR,
    ensure_dirs,
)
from src.agent.search_agent import SearchAgent
from src.evaluation.metrics import evaluate_example_metrics
from src.judge.judges import OracleJudge, ProcessJudge
from src.llm.gemini_client import GeminiClient
from src.tools.chunk_quality import QUALITY_SCORES_PATH
from src.tools.retrieval_tools import ToolRegistry


@dataclass
class ExperimentCondition:
    """Single experiment condition specification."""

    suite: str
    retriever: str
    max_steps: int
    process_feedback: bool
    quality_reweight: bool
    judge_mode: str  # "none", "process", "oracle"

    @property
    def condition_id(self) -> str:
        return (
            f"{self.suite}__{self.retriever}"
            f"__steps{self.max_steps}"
            f"__judge{int(self.process_feedback)}"
            f"__quality{int(self.quality_reweight)}"
            f"__mode{self.judge_mode}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic RAG experiments")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Smoke = small subset for correctness checks, full = entire QA set.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=3,
        help="Question count for smoke mode; ignored in full mode unless --limit is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard cap on number of QA examples after sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic QA subset sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Chunks retrieved per step.",
    )
    parser.add_argument(
        "--max-steps-values",
        type=str,
        default="2,3,4",
        help="Comma-separated max_steps values for main grid.",
    )
    parser.add_argument(
        "--run-main-grid",
        action="store_true",
        help="Run primary 36-condition grid.",
    )
    parser.add_argument(
        "--only-quality-off",
        action="store_true",
        help="For main grid runs, keep only conditions with quality_reweight=0.",
    )
    parser.add_argument(
        "--run-oracle-mini-study",
        action="store_true",
        help="Run separate oracle mini-study on smoke subset.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional label in output directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory path. Defaults to data/results/<timestamp>_<tag>.",
    )
    parser.add_argument(
        "--qa-concurrency",
        type=int,
        default=GEMINI_QA_CONCURRENCY_DEFAULT,
        help=(
            "Max concurrent QA trajectories within a single condition (default from "
            "GEMINI_QA_CONCURRENCY_DEFAULT). Higher overlaps local retrieval work; "
            "very high values vs GEMINI_CONCURRENT_LIMIT increase API contention tails."
        ),
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=GEMINI_EVAL_CONCURRENCY,
        help=(
            "Max concurrent post-hoc LLM evaluator calls (global within this process). "
            "Caps 'thundering herd' when many agents finish together (default from "
            "GEMINI_EVAL_CONCURRENCY)."
        ),
    )
    parser.add_argument(
        "--condition-concurrency",
        type=int,
        default=1,
        help="Max concurrent conditions (within one process). "
             "Usually leave at 1 and shard across processes for real parallelism.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard selector 'i/N' — run only conditions where index %% N == i. "
             "Use to split the grid across multiple processes with separate quotas.",
    )
    return parser.parse_args()


def usage_snapshot(client: GeminiClient) -> dict:
    stats = client.stats
    return {
        "requests": stats.requests,
        "errors": stats.errors,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "embedding_tokens": stats.embedding_tokens,
        "estimated_cost": stats.estimated_cost,
    }


def usage_delta(before: dict, after: dict) -> dict:
    return {
        "requests": after["requests"] - before["requests"],
        "errors": after["errors"] - before["errors"],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "embedding_tokens": after["embedding_tokens"] - before["embedding_tokens"],
        "estimated_cost": round(after["estimated_cost"] - before["estimated_cost"], 8),
    }


def build_main_conditions(
    max_steps_values: list[int],
    only_quality_off: bool = False,
) -> list[ExperimentCondition]:
    conditions: list[ExperimentCondition] = []
    for retriever in ["grep", "bm25", "embedding"]:
        for steps in max_steps_values:
            for process_feedback in [False, True]:
                for quality_reweight in [False, True]:
                    if only_quality_off and quality_reweight:
                        continue
                    conditions.append(
                        ExperimentCondition(
                            suite="main",
                            retriever=retriever,
                            max_steps=steps,
                            process_feedback=process_feedback,
                            quality_reweight=quality_reweight,
                            judge_mode="process" if process_feedback else "none",
                        )
                    )
    return conditions


def build_oracle_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(
            suite="oracle-mini",
            retriever="bm25",
            max_steps=3,
            process_feedback=False,
            quality_reweight=False,
            judge_mode="none",
        ),
        ExperimentCondition(
            suite="oracle-mini",
            retriever="bm25",
            max_steps=3,
            process_feedback=True,
            quality_reweight=False,
            judge_mode="process",
        ),
        ExperimentCondition(
            suite="oracle-mini",
            retriever="bm25",
            max_steps=3,
            process_feedback=True,
            quality_reweight=False,
            judge_mode="oracle",
        ),
    ]


def apply_shard(
    conditions: list[ExperimentCondition],
    shard: str | None,
) -> list[ExperimentCondition]:
    """Filter conditions by 'i/N' shard selector for multi-process runs."""
    if not shard:
        return conditions
    try:
        i_str, n_str = shard.split("/")
        i, n = int(i_str), int(n_str)
    except ValueError as e:
        raise ValueError(f"Invalid --shard value '{shard}'; expected 'i/N'.") from e
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"Invalid shard indices in '{shard}'; need 0 <= i < N.")
    return [c for idx, c in enumerate(conditions) if idx % n == i]


def load_qa_subset(mode: str, num_questions: int, seed: int, limit: int | None) -> list[dict]:
    with open(QA_PAIRS_JSON) as f:
        qa_pairs = json.load(f)

    if mode == "smoke":
        rng = random.Random(seed)
        sample_size = min(num_questions, len(qa_pairs))
        qa_pairs = rng.sample(qa_pairs, sample_size)

    if limit is not None:
        qa_pairs = qa_pairs[:limit]
    return qa_pairs


def create_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = []
        if args.output_tag:
            suffix_parts.append(args.output_tag)
        if args.shard:
            # Make shard directories distinct so parallel processes don't collide.
            suffix_parts.append(f"shard{args.shard.replace('/', '-of-')}")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        out_dir = RESULTS_DIR / f"{timestamp}{suffix}"
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    return out_dir


async def append_jsonl(path: Path, payload: dict, lock: asyncio.Lock) -> None:
    """Append to JSONL with an async lock so concurrent writers don't interleave lines."""
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    async with lock:
        # File I/O is sync; under the lock it's fine and each write is tiny.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "success",
        "judge_score",
        "retrieval_precision",
        "hallucination_rate",
        "lexical_f1",
        "exact_match",
        "reference_episode_recall",
        "trajectory_length",
        "elapsed_seconds",
        "cost_usd",
    ]
    group_cols = [
        "suite",
        "condition_id",
        "retriever",
        "max_steps",
        "process_feedback",
        "quality_reweight",
        "judge_mode",
        "qa_count",
    ]
    summary = (
        df.groupby(group_cols, as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["suite", "condition_id"])
    )
    return summary


async def _run_single_qa(
    qa_idx: int,
    qa: dict,
    condition: ExperimentCondition,
    client: GeminiClient,
    registry: ToolRegistry,
    output_dir: Path,
    top_k: int,
    qa_count: int,
    runs_jsonl_lock: asyncio.Lock,
    sem: asyncio.Semaphore,
    eval_sem: asyncio.Semaphore,
) -> dict:
    """Run one (condition, question) trajectory end-to-end. Safe to run concurrently."""
    async with sem:
        question = qa["question"]
        answer = qa["answer"]
        reference_episodes = qa.get("reference_episodes", [])
        qa_type = qa.get("qa_type", "unknown")
        era = qa.get("era", "unknown")
        qa_id = f"q{qa_idx:03d}"

        if condition.judge_mode == "process":
            judge = ProcessJudge(client=client)
        elif condition.judge_mode == "oracle":
            judge = OracleJudge(client=client, ground_truth=answer)
        else:
            judge = None

        agent = SearchAgent(
            client=client,
            tool_registry=registry,
            tool_name=condition.retriever,
            max_steps=condition.max_steps,
            top_k_per_step=top_k,
            judge=judge,
        )

        # Note: usage_snapshot/delta is best-effort under concurrency — GeminiClient
        # stats are process-global, so deltas around concurrent trajectories will
        # include other in-flight work. The aggregate totals at end-of-run are exact.
        before = usage_snapshot(client)
        trajectory = await agent.run(question)
        after = usage_snapshot(client)
        delta = usage_delta(before, after)

        trajectory_dict = trajectory.to_dict()

    trajectory_path = (
        output_dir
        / "trajectories"
        / f"{condition.condition_id}__{qa_id}.json"
    )
    with open(trajectory_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_dict, f, indent=2, ensure_ascii=True)

    async with eval_sem:
        metrics = await evaluate_example_metrics(
            client=client,
            question=question,
            ground_truth=answer,
            predicted_answer=trajectory.final_answer,
            reference_episodes=reference_episodes,
            trajectory=trajectory_dict,
        )

    record = {
        "suite": condition.suite,
        "condition_id": condition.condition_id,
        "retriever": condition.retriever,
        "max_steps": condition.max_steps,
        "process_feedback": condition.process_feedback,
        "quality_reweight": condition.quality_reweight,
        "judge_mode": condition.judge_mode,
        "qa_id": qa_id,
        "qa_type": qa_type,
        "era": era,
        "question": question,
        "ground_truth_answer": answer,
        "predicted_answer": trajectory.final_answer,
        "reference_episodes": reference_episodes,
        "trajectory_path": str(trajectory_path),
        "qa_count": qa_count,
        "trajectory_length": len(trajectory.steps),
        "elapsed_seconds": trajectory.elapsed_seconds,
        "total_chunks_retrieved": trajectory.total_chunks_retrieved,
        "usage_delta": delta,
        "cost_usd": delta["estimated_cost"],
        **metrics.to_dict(),
    }
    await append_jsonl(output_dir / "runs.jsonl", record, runs_jsonl_lock)
    return record


async def run_single_condition(
    condition: ExperimentCondition,
    qa_items: list[dict],
    client: GeminiClient,
    output_dir: Path,
    top_k: int,
    qa_concurrency: int,
    runs_jsonl_lock: asyncio.Lock,
    eval_sem: asyncio.Semaphore,
) -> list[dict]:
    print(
        f"\n=== Running {condition.condition_id} on {len(qa_items)} questions "
        f"(qa_concurrency={qa_concurrency}) ==="
    )

    if condition.quality_reweight and not QUALITY_SCORES_PATH.exists():
        print(
            "WARNING: quality_reweight enabled but chunk quality score file not found. "
            "Reweighting will effectively be disabled."
        )

    registry = ToolRegistry(
        gemini_client=client,
        quality_reweight=condition.quality_reweight,
    )

    sem = asyncio.Semaphore(qa_concurrency)
    progress_desc = (
        f"{condition.retriever}|steps={condition.max_steps}"
        f"|judge={condition.judge_mode}|quality={int(condition.quality_reweight)}"
    )

    tasks = [
        _run_single_qa(
            qa_idx=qa_idx,
            qa=qa,
            condition=condition,
            client=client,
            registry=registry,
            output_dir=output_dir,
            top_k=top_k,
            qa_count=len(qa_items),
            runs_jsonl_lock=runs_jsonl_lock,
            sem=sem,
            eval_sem=eval_sem,
        )
        for qa_idx, qa in enumerate(qa_items)
    ]

    # tqdm_asyncio.gather preserves submission order in the returned list.
    records: list[dict] = await tqdm_asyncio.gather(
        *tasks,
        desc=progress_desc,
        leave=False,
    )
    return records


async def run_condition_group(
    conditions: list[ExperimentCondition],
    qa_items: list[dict],
    client: GeminiClient,
    output_dir: Path,
    top_k: int,
    qa_concurrency: int,
    condition_concurrency: int,
    runs_jsonl_lock: asyncio.Lock,
    eval_sem: asyncio.Semaphore,
    group_desc: str,
) -> list[dict]:
    """Run a list of conditions, optionally with N conditions in flight at once."""
    if condition_concurrency <= 1:
        # Sequential path — preserves original behavior and progress bar semantics.
        all_records: list[dict] = []
        for condition in tqdm(conditions, desc=group_desc):
            all_records.extend(
                await run_single_condition(
                    condition=condition,
                    qa_items=qa_items,
                    client=client,
                    output_dir=output_dir,
                    top_k=top_k,
                    qa_concurrency=qa_concurrency,
                    runs_jsonl_lock=runs_jsonl_lock,
                    eval_sem=eval_sem,
                )
            )
        return all_records

    cond_sem = asyncio.Semaphore(condition_concurrency)

    async def _run(condition: ExperimentCondition) -> list[dict]:
        async with cond_sem:
            return await run_single_condition(
                condition=condition,
                qa_items=qa_items,
                client=client,
                output_dir=output_dir,
                top_k=top_k,
                qa_concurrency=qa_concurrency,
                runs_jsonl_lock=runs_jsonl_lock,
                eval_sem=eval_sem,
            )

    nested: list[list[dict]] = await tqdm_asyncio.gather(
        *[_run(c) for c in conditions],
        desc=group_desc,
    )
    flat: list[dict] = []
    for chunk in nested:
        flat.extend(chunk)
    return flat


async def main() -> None:
    args = parse_args()
    if not args.run_main_grid and not args.run_oracle_mini_study:
        # Default behavior for convenience.
        args.run_main_grid = True

    ensure_dirs()
    qa_items = load_qa_subset(
        mode=args.mode,
        num_questions=args.num_questions,
        seed=args.seed,
        limit=args.limit,
    )
    if not qa_items:
        raise ValueError("No QA items available for experiment run.")

    max_steps_values = [int(x.strip()) for x in args.max_steps_values.split(",") if x.strip()]
    if len(max_steps_values) != 3:
        print("WARNING: expected 3 max_steps values for strict 36 grid.")

    if args.qa_concurrency > GEMINI_CONCURRENT_LIMIT * 4:
        print(
            f"WARNING: --qa-concurrency={args.qa_concurrency} is high vs "
            f"GEMINI_CONCURRENT_LIMIT={GEMINI_CONCURRENT_LIMIT}; expect longer tail latency / retries."
        )
    if args.condition_concurrency > 1 and args.eval_concurrency > GEMINI_CONCURRENT_LIMIT * 2:
        print(
            "WARNING: --condition-concurrency > 1 with high --eval-concurrency can spike API load; "
            "consider lowering eval concurrency or sharding across processes."
        )

    print(
        f"Tuning: RPM={GEMINI_RPM_LIMIT} (burst cap={GEMINI_RPM_BURST_CAPACITY}), "
        f"concurrent_sdk={GEMINI_CONCURRENT_LIMIT}, "
        f"qa_concurrency={args.qa_concurrency}, eval_concurrency={args.eval_concurrency}, "
        f"condition_concurrency={args.condition_concurrency}"
    )

    output_dir = create_output_dir(args)
    print(f"Writing outputs to: {output_dir}")
    if args.shard:
        print(f"Running shard {args.shard}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "qa_count": len(qa_items),
        "qa_file": str(QA_PAIRS_JSON),
        "quality_scores_path": str(QUALITY_SCORES_PATH),
    }
    # vars(args) contains a Path; make it JSON-serializable.
    manifest["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in manifest["args"].items()}
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    client = GeminiClient()
    runs_jsonl_lock = asyncio.Lock()
    eval_sem = asyncio.Semaphore(args.eval_concurrency)
    all_records: list[dict] = []

    if args.run_main_grid:
        main_conditions = build_main_conditions(
            max_steps_values=max_steps_values,
            only_quality_off=args.only_quality_off,
        )
        main_conditions = apply_shard(main_conditions, args.shard)
        if not main_conditions:
            print("Main grid has no conditions after shard filter; skipping.")
        else:
            with open(output_dir / "conditions_main.json", "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(c) | {"condition_id": c.condition_id} for c in main_conditions],
                    f,
                    indent=2,
                )
            all_records.extend(
                await run_condition_group(
                    conditions=main_conditions,
                    qa_items=qa_items,
                    client=client,
                    output_dir=output_dir,
                    top_k=args.top_k,
                    qa_concurrency=args.qa_concurrency,
                    condition_concurrency=args.condition_concurrency,
                    runs_jsonl_lock=runs_jsonl_lock,
                    eval_sem=eval_sem,
                    group_desc="Main grid conditions",
                )
            )

    if args.run_oracle_mini_study:
        oracle_conditions = build_oracle_conditions()
        oracle_conditions = apply_shard(oracle_conditions, args.shard)
        if not oracle_conditions:
            print("Oracle mini study has no conditions after shard filter; skipping.")
        else:
            with open(output_dir / "conditions_oracle_mini.json", "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(c) | {"condition_id": c.condition_id} for c in oracle_conditions],
                    f,
                    indent=2,
                )
            all_records.extend(
                await run_condition_group(
                    conditions=oracle_conditions,
                    qa_items=qa_items,
                    client=client,
                    output_dir=output_dir,
                    top_k=args.top_k,
                    qa_concurrency=args.qa_concurrency,
                    condition_concurrency=args.condition_concurrency,
                    runs_jsonl_lock=runs_jsonl_lock,
                    eval_sem=eval_sem,
                    group_desc="Oracle mini conditions",
                )
            )

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(output_dir / "per_example_metrics.csv", index=False)
        summary = aggregate_results(df)
        summary.to_csv(output_dir / "summary_by_condition.csv", index=False)

        overview = {
            "rows": len(df),
            "conditions": int(df["condition_id"].nunique()),
            "qa_count_per_condition": int(df["qa_count"].max()),
            "overall_success_rate": float(df["success"].mean()),
            "overall_judge_score": float(df["judge_score"].mean()),
            "overall_retrieval_precision": float(df["retrieval_precision"].mean()),
            "overall_hallucination_rate": float(df["hallucination_rate"].mean()),
            "overall_cost_usd": float(df["cost_usd"].sum()),
        }
        with open(output_dir / "summary_overview.json", "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2, ensure_ascii=True)

    with open(output_dir / "gemini_usage_final.json", "w", encoding="utf-8") as f:
        json.dump(usage_snapshot(client), f, indent=2, ensure_ascii=True)
    client.print_stats()
    print("Experiment run complete.")


if __name__ == "__main__":
    asyncio.run(main())