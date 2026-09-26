# Results artifacts

This directory is for **full-benchmark** outputs only. Do not commit numbers or figures from toy data, smoke tests, or partial dataset mirrors as if they were AqSolDB benchmark results.

After running the full 9,982-row source dataset through cleaning, splitting, baseline training, GNN training, and report generation, the expected review artifacts are:

- `tables/benchmark_summary.csv`
- `tables/benchmark_results_long.csv`
- `tables/applicability_summary.csv`
- `tables/failure_cases.csv`
- `figures/parity_random.png`
- `figures/parity_scaffold.png`
- `figures/residuals_random.png`
- `figures/residuals_scaffold.png`
- `figures/applicability_random.png`
- `figures/applicability_scaffold.png`
- `figures/failure_molecules_scaffold.png`
- `report.md`

The report is regenerated from saved per-molecule predictions rather than copied from training summaries. This is intentional: it makes the prediction rows the auditable source of truth for final metrics.
