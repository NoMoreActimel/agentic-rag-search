# Era-stratified analysis — does agentic RAG suffer from the knowledge cutoff?

**Run:** `data/results/submit87_fast_merged` (87 QA × 36 grid conditions = 3132 rows)
**Cohorts:** every QA pair is tagged by its reference-episode era relative to
Gemini 2.5 Flash's 2025-01-31 training cutoff:

- `old`   — all refs pre-cutoff (model saw them in pretraining). `n=32`
- `new`   — all refs post-cutoff (unseen by the model). `n=23`
- `mixed` — refs span both eras. `n=32`

**Method.** For each question we first average the six metrics across the 36
grid conditions (so every question contributes one number), then aggregate
within each era. 95% CIs are non-parametric bootstraps over questions
(`n_boot=5000`). Old-vs-new is tested with Mann-Whitney U on per-question
means. Artifacts: `era_means.csv`, `era_stat_tests.csv`,
`era_by_retriever.csv`, `era_by_qa_type.csv`.

## Headline — the hypothesis is *not* supported

If the agent were leaning on parametric memory, we would expect `old` to beat
`new`. We see the opposite on every quality metric:

| Metric                    |   old |   mixed |   new | Δ (new−old) | MWU p |
|---------------------------|------:|--------:|------:|------------:|------:|
| Success rate              | 0.717 |   0.759 | 0.802 |      +0.085 | 0.099 |
| LLM judge (1–5)           | 4.106 |   4.176 | 4.314 |      +0.208 | 0.129 |
| Reference episode recall  | 0.541 |   0.648 | 0.682 |      +0.141 | **0.019** |
| Retrieval precision       | 0.484 |   0.467 | 0.501 |      +0.017 | 0.676 |
| Hallucination rate (↓)    | 0.023 |   0.038 | 0.027 |      +0.004 | 0.762 |
| Lexical F1                | 0.405 |   0.390 | 0.368 |      −0.037 | 0.059 |

Figure: `era_overview.png`. See the sibling CSVs for the raw numbers.

**Read.** End-task quality on post-cutoff questions is **as good or slightly
better** than on pre-cutoff questions. The one statistically significant gap
is reference-episode recall — the retriever finds the gold episodes *more
reliably* for new QA.

Note on mechanism: retrieval is over the **full shared index** in every call
(`src/tools/retrieval_tools.py` — no per-query era gate). What differs is
distractor density. The index contains **393** pre-cutoff episodes vs **33**
post-cutoff ones (58,571 vs 7,769 chunks). A `new` QA's 2–3 gold episodes
compete against ~30 same-era distractors; an `old` QA's gold episodes compete
against ~390. Post-cutoff content is also topically distinctive (recent 2025
events/guests), so it collides less with the large pre-cutoff backdrop. That
is a retrieval-side effect, not a memorization-side one.

The only metric that leans toward `old` is lexical F1 (marginal, p=0.06),
which is a surface-overlap metric vulnerable to phrasing drift and is not a
faithfulness signal on its own — every other measure goes the other way.
Hallucination rate is essentially flat across eras.

## Where the effect shows up

- **By retriever** (`era_by_retriever.png`). The new ≥ old pattern holds across
  grep, BM25, and embedding. The old-vs-new gap is biggest for grep
  (success 0.656 → 0.797), where retrieval quality dominates — again pointing
  at retrieval, not memorization, as the driver. Embedding is the most
  era-robust retriever.
- **By qa_type** (`era_by_qa_type.png`). The new ≥ old pattern is
  type-dependent: comparative and aggregation questions are easier in the
  `new` cohort, multihop is roughly flat, and temporal-old is the hardest
  cell in the grid (success 0.552, n=8). Temporal-new (n=3) should be
  interpreted with caution — the post-cutoff pool is small.

## Caveats

- Cohort sizes are uneven (`old`/`mixed` n=32, `new` n=23). The bootstrap CIs
  reflect this; `new` intervals are wider.
- `qa_type` composition differs across eras (`temporal`/`aggregation` are
  under-represented in `new`). The macro-averaged table
  `era_means_macro_by_qa_type.csv` — which weights every qa_type equally
  within each era — gives 0.717 / 0.759 / 0.794 success for old/mixed/new, so
  the ordering is not an artifact of qa_type mix.
- `success` here is the condition-level binary success flag averaged per QA.
  The judge score uses the same LLM family as the agent; it is not an
  independent adjudicator. This is a shared caveat across the whole benchmark,
  not an era-specific one.

## Bottom line for the write-up

On this benchmark, agentic RAG **does not appear to be bottlenecked by the
knowledge cutoff**. The agent's answer quality tracks what the retriever
surfaces, and retrieval happens to be slightly *better* on the post-cutoff
cohort because that subcorpus is smaller. If there is a cutoff penalty, it is
small enough to be swamped by retrieval-side effects at the current sample
size — the strongest era-robust retriever in the grid is the embedding one.
