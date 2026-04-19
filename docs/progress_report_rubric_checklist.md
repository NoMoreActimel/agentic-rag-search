# Progress Report Rubric Checklist

This checklist maps your stated rubric to concrete sections and assets in the drafted report.

Primary draft:

- `docs/progress_report_acl_overleaf_draft.tex`

Evidence bundle:

- `data/results/20260330_185830_report_full_qualityoff/`
- `data/results/20260330_185830_report_full_qualityoff/analysis/tables/`
- `data/results/20260330_185830_report_full_qualityoff/analysis/plots/`
- `data/results/20260330_185830_report_full_qualityoff/analysis/plots/report/`

---

## 1) Self-contained report (1 point)

- Covered in abstract + introduction + setup sections of the draft.
- Proposal context is restated in report text without requiring proposal reading.
- Scope constraints (quality-off completed, quality-on pending) are explicitly stated.

Status: **Satisfied**

---

## 2) Appropriate empirical progress for elapsed time (3 points)

- Completed full quality-off run:
  - 18 conditions x 20 QA each = 360 evaluations.
- Reproducible artifact set generated and analyzed.
- Automated evaluation scripts and analysis pipeline are in place.

Status: **Satisfied** (with explicit note that quality-on remains pending)

---

## 3) Experimental methods clearly described (2 points)

- Models and systems:
  - retrieval backends (`grep`, `bm25`, `embedding`)
  - process feedback on/off
  - step budgets `{2,3,4}`
- Hyperparameter-level details included:
  - `top_k=5`, 20-item full run, condition counts.
- Data and run provenance:
  - exact result directory and manifests referenced.

Status: **Satisfied**

---

## 4) Quantitative results clearly presented (2 points)

- Figures prioritized over large tables.
- Confidence intervals added where appropriate in report-focused plots:
  - `fig_success_by_retriever_ci.png`
  - `fig_steps_vs_success_ci.png`
  - `fig_success_by_qatype_ci.png`
- Axes and labels are explicit; clutter controlled via 3-up layout and focused figure set.
- Compact table retained only for exact KPI values.

Status: **Satisfied**

---

## 5) Analysis of results and preliminary error analysis (2 points)

- Interpretation includes:
  - retriever vs outcome mismatch (precision vs success),
  - process-feedback interaction effects,
  - step-budget quality/cost tradeoff.
- Preliminary error analysis includes:
  - hardest question family (aggregation),
  - per-question variability visualization and discussion.

Status: **Satisfied**

---

## 6) Clear next steps to end of semester (2 points)

- Next steps section includes:
  1. complete chunk-quality scoring with failed-row hygiene,
  2. full quality-on rerun,
  3. process-feedback ablations,
  4. deeper qualitative trajectory analysis.

Status: **Satisfied**

---

## Final pre-submission checks (recommended)

1. Compile `progress_report_acl_overleaf_draft.tex` in Overleaf and verify page length.
2. Ensure all six figure files are uploaded and render clearly.
3. Keep one consistent cost convention in prose (client-wide vs per-example aggregate), with one footnote explaining the difference.
4. Keep quality-on claims strictly in future-work language.

