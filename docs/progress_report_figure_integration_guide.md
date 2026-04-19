# Progress Report Figure Integration Guide

This guide maps exact figure files to the ACL draft in:
`docs/progress_report_acl_overleaf_draft.tex`.

## 1) Copy these images into your Overleaf project

From:
`data/results/20260330_185830_report_full_qualityoff/analysis/plots/report/`

- `fig_success_by_retriever_ci.png`
- `fig_process_feedback_delta_heatmap.png`
- `fig_steps_vs_success_ci.png`
- `fig_success_by_qatype_ci.png`
- `fig_cost_success_frontier.png`

From:
`data/results/20260330_185830_report_full_qualityoff/analysis/plots/`

- `question_condition_success_heatmap.png`

## 2) Where each figure goes

### Figure block A (`fig:main_three`)
Use these 3 images side-by-side:
1. `fig_success_by_retriever_ci.png`
2. `fig_process_feedback_delta_heatmap.png`
3. `fig_steps_vs_success_ci.png`

Purpose:
- covers retriever comparison, process-feedback interaction, and step-budget trend with uncertainty bars.

### Figure block B (`fig:error_tradeoff`)
Use these 3 images side-by-side:
1. `fig_success_by_qatype_ci.png`
2. `fig_cost_success_frontier.png`
3. `question_condition_success_heatmap.png`

Purpose:
- shows question-family difficulty, efficiency frontier, and per-question heterogeneity.

## 3) Recommended visual checks before PDF export

- Ensure no axis labels are clipped after Overleaf compile.
- Keep image width settings exactly as in draft (`0.32\linewidth` each in 3-up rows).
- Verify font remains legible at normal page view (no zoom required).
- If one subplot looks dense, reduce tick labels (especially heatmap x-axis) rather than shrinking the whole figure.

## 4) Optional fallback if page space is tight

Priority order (keep highest first):
1. `fig_process_feedback_delta_heatmap.png`
2. `fig_steps_vs_success_ci.png`
3. `fig_success_by_retriever_ci.png`
4. `fig_success_by_qatype_ci.png`
5. `fig_cost_success_frontier.png`
6. `question_condition_success_heatmap.png`

If you must drop one, drop the heatmap in block B first (`question_condition_success_heatmap.png`) and keep a short sentence on per-question variance in text.
