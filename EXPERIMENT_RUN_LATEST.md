# Experiment run — full main grid (Vertex v2)

This document summarizes how the **full 36-condition × 20 QA** experiment completed, where outputs live, integrity checks, metrics, and how to interpret cost/error accounting. Written for a clean handoff (e.g. push to repo and share).

**Primary results directory:**

`agentic-rag-search/data/results/20260420_221641_full_main36_vertex_v2/`

---

## 1. Terminal evidence (orchestration)

From the run log (tmux / terminal capture):

- Progress reached **36/36** main-grid conditions; final cells included **embedding**, **steps 4**, **judge on/off**, **quality on/off**, **process** mode — consistent with the intended **36-cell** grid.
- Final usage line reported:
  - **Backend:** Vertex
  - **Requests:** 4918  
  - **Errors:** 16  
  - **Input / output / embedding tokens:** ~8.69M / ~465.7k / 5385  
  - **Estimated cost:** ~**$1.0553**  
  - **Client stats elapsed:** ~**30586.4 s** (~8.5 h on the `GeminiClient` usage timer)
- Message **`Experiment run complete.`** indicates normal shutdown (not a mid-run hang).

**Conclusion:** At the orchestration level the run **finished successfully**.

---

## 2. Artifact inventory

| File / folder | Purpose |
|---------------|---------|
| `manifest.json` | Run metadata, CLI args snapshot, `qa_count`, paths to QA and chunk quality scores |
| `conditions_main.json` | The 36 conditions written before the grid loop |
| `runs.jsonl` | One JSON object per **(condition, question)** |
| `per_example_metrics.csv` | Same rows as JSONL, tabular for analysis |
| `summary_by_condition.csv` | Per-condition means |
| `summary_overview.json` | Global aggregates |
| `gemini_usage_final.json` | Final `GeminiClient` usage snapshot (matches terminal summary) |
| `trajectories/*.json` | One trajectory JSON per example (**720** files) |

Approximate sizes: CSV ~1.5 MB, JSONL ~1.9 MB — consistent with a full run.

---

## 3. Integrity checklist

| Check | Result |
|--------|--------|
| Rows in `per_example_metrics.csv` (use **pandas** row count; `wc -l` can mis-count if fields contain newlines) | **720** |
| Lines in `runs.jsonl` | **720** |
| Unique `condition_id` | **36** |
| Rows per condition | **20** each (min = max = 20) |
| Duplicate (`condition_id`, `qa_id`) | **0** |
| IDs in `conditions_main.json` vs CSV | **Match** |
| Trajectory files | **720**; spot-checks: all `{condition_id}__{qa_id}.json` present |

**`summary_overview.json`** matches recomputation from the CSV for `rows`, overall success rate, and sum of `cost_usd`.

---

## 4. Manifest nuance (`num_questions` vs `qa_count`)

`manifest.json` may show `args.num_questions: 3` because that is the CLI **default**. For **`mode: full`**, `--num-questions` is **ignored** unless `--limit` is set (see `scripts/05_run_experiments.py` help text). The authoritative field for how many questions ran is **`qa_count`** (here **20**).

---

## 5. Grid coverage (factor levels)

Verified balances:

- **`quality_reweight`:** 360 `false`, 360 `true`
- **`process_feedback`:** 360 / 360
- **`retriever`:** 240 each — `grep`, `bm25`, `embedding`
- **`max_steps`:** 240 each — 2, 3, 4

This is the full **3 × 3 × 2 × 2 × 2** design (36 configs × 20 QAs = **720** rows).

---

## 6. Global metrics (from CSV / `summary_overview.json`)

- **Mean success (`success`):** **0.45**
- **Mean judge score:** **~3.13** (scale 1–5)
- **Mean retrieval precision:** **~0.33**
- **Mean hallucination rate:** **~0.31**
- **Exact match:** all **false** in this run
- **Lexical F1:** populated for all rows (rough range ~0.15–0.63 in spot checks)
- **Sum of per-row `cost_usd`:** **~0.9122 USD** (matches `summary_overview.json`)

**Per-example `elapsed_seconds`** (agent trajectory timing): wide spread (on the order of seconds to many minutes for worst cases); mean ~**26 s** — this is **not** the same clock as the ~30586 s **client usage** timer in the terminal.

**Evaluator health:**

- **0** rows with `"Evaluator failed"` in `reason`
- **0** rows with `"Unable to generate"` in predicted answers
- **65** rows with `judge_score == 1` (harsh outcomes or failures worth qualitative sampling)

---

## 7. Quality reweight ON vs OFF (paired by design)

Aggregate split (360 rows each):

| Slice | Mean success | Mean judge | Mean retrieval P | Mean hallucination | Sum `cost_usd` |
|--------|----------------|------------|------------------|--------------------|----------------|
| `quality_reweight=false` | **~0.461** | **~3.22** | **~0.334** | **~0.299** | **~0.454** |
| `quality_reweight=true` | **~0.439** | **~3.04** | **~0.327** | **~0.330** | **~0.458** |

**Paired** comparison (same `retriever`, `max_steps`, judge bit from `condition_id`, `process_feedback`, `qa_id`; only quality differs): **360** pairs.

- Mean **Δ success** (on − off): **~−0.022**
- **Wins** (quality on better / tie / quality off better): **60 / 232 / 68**
- Mean **Δ judge_score** (on − off): **~−0.175**
- Mean **Δ hallucination** (on − off): **~+0.031** (on reports higher hallucination on average)

**Interpretation:** Under this run’s chunk-scoring and reweighting setup, aggregate metrics **did not improve** with quality reweight; they moved slightly **the wrong way** on success, judge, and hallucination. Any paper claim should stay scoped to **this** implementation and benchmark.

---

## 8. Best / worst conditions (high level)

From `summary_by_condition.csv` (means):

- **Lowest success (~0.30):** several **embedding** configurations (e.g. low steps + judge off + quality off, and some process-judge variants).
- **Highest success (~0.60–0.70):** **grep**-heavy configs (e.g. `main__grep__steps4__judge0__quality0__modenone` near **0.70**).

Suggests **grep** matched this QA set’s structure better than dense retrieval for many items (worth deeper breakdown by QA type if needed).

---

## 9. Reconciling errors: 16 global vs 14 in summed `usage_delta`

**`gemini_usage_final.json`:** **16** errors (matches terminal).

**Per-row `usage_delta.errors` in `runs.jsonl`:**

- **11** rows have `usage_delta.errors > 0`
- **Sum** of `usage_delta.errors` over all **720** rows = **14**

**Why the gap**

In `run_single_condition` (`scripts/05_run_experiments.py`), usage is snapshotted **before** `agent.run` and **after** `agent.run`, **before** `evaluate_example_metrics`. The evaluator calls the same `GeminiClient`; each failed attempt increments `stats.errors` even if retries eventually succeed.

So:

- **`usage_delta.errors`** ≈ failures (including retries) during **search / generation inside the agent loop**.
- **Extra global errors** (here **16 − 14 = 2**) fit **evaluator-side** failed attempts that occurred **after** the post–`agent.run` snapshot — e.g. retries that then succeeded, which is consistent with **zero** `"Evaluator failed"` rows in the CSV.

---

## 10. Reconciling cost: ~$1.055 vs ~$0.912

- Per-row **`cost_usd`** is derived from **`usage_delta["estimated_cost"]`**, i.e. cost accumulated only between the **before** and **after** snapshots around **`agent.run`**. It does **not** include the separate **LLM judge / metrics** call in `evaluate_example_metrics`.
- **`gemini_usage_final.json`** (and the terminal line) reflect **cumulative** client usage for the **entire** run (agent + evaluator + embeddings).

Both numbers are **correct** once that split is understood. If you need per-row **total** cost including evaluation, the experiment script would need an extra snapshot after `evaluate_example_metrics` (future improvement).

---

## 11. API / reliability verdict

- **16** errors over **4918** requests is a **low** rate; behavior is consistent with **transient** issues handled by retries.
- No evidence of mass evaluator failure or missing trajectories for completed rows.

---

## 12. Related code paths (for your friend)

| Topic | Location |
|--------|----------|
| Experiment driver | `agentic-rag-search/scripts/05_run_experiments.py` |
| Usage snapshots / delta | Same file: `usage_snapshot`, `usage_delta`, `run_single_condition` |
| Gemini client, errors, retries | `agentic-rag-search/src/llm/gemini_client.py` |
| Post-hoc LLM metrics | `agentic-rag-search/src/evaluation/metrics.py` — `evaluate_example_metrics` |
| Chunk quality / reweight | `agentic-rag-search/src/tools/chunk_quality.py` + `data/processed/chunk_quality_scores.json` |

---

## 13. Quick verification commands

From `agentic-rag-search/` with the project venv:

```bash
python3 -c "import pandas as pd; df=pd.read_csv('data/results/20260420_221641_full_main36_vertex_v2/per_example_metrics.csv'); print(len(df), df['condition_id'].nunique())"
```

Expect: `720` and `36`.

```bash
wc -l data/results/20260420_221641_full_main36_vertex_v2/runs.jsonl
```

Expect: `720` lines.

---

*Generated for handoff: full main grid Vertex v2 run (`20260420_221641_full_main36_vertex_v2`).*
