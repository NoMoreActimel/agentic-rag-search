#!/usr/bin/env python3
"""Launch multiple 05_run_experiments.py processes with disjoint --shard i/N selectors.

Example:
  .venv-linux/bin/python scripts/05b_launch_shards.py --shards 3 -- \\
    --run-main-grid --mode full --max-steps-values 2,3,4 --top-k 5 --output-tag myrun

Each worker gets --shard i/3 appended. Output dirs default to timestamped paths with
shard suffix (see 05_run_experiments.ensure_output_dir).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N experiment shards in parallel.")
    parser.add_argument(
        "--shards",
        type=int,
        required=True,
        help="Number of parallel workers (shard index runs 0 .. N-1).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current).",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "05_run_experiments.py",
        help="Path to 05_run_experiments.py.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to each worker (use '--' before flags, e.g. -- --run-main-grid ...).",
    )
    args = parser.parse_args()
    n = args.shards
    if n <= 0:
        raise SystemExit("--shards must be positive")

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    script = str(args.script.resolve())
    procs: list[subprocess.Popen] = []
    for i in range(n):
        shard = f"{i}/{n}"
        cmd = [args.python, script, *extra, "--shard", shard]
        print(f"[launcher] starting shard {shard}: {' '.join(cmd)}")
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        )

    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        print(f"[launcher] one or more shards failed: exit codes={codes}")
        raise SystemExit(1)
    print("[launcher] all shards finished successfully.")


if __name__ == "__main__":
    main()
