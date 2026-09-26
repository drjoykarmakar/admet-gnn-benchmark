#!/usr/bin/env python
"""Regenerate benchmark tables, figures, and failure-case summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    applicability_summary,
    attach_applicability_domain,
    benchmark_results,
    nearest_train_tanimoto,
    select_failure_cases,
    wide_benchmark_table,
)
from src.plots import (  # noqa: E402
    save_applicability_plot,
    save_failure_molecule_grid,
    save_gnn_learning_curve,
    save_parity_plot,
    save_residual_plot,
)


PREDICTION_FILES = {
    ("baseline", "random"): "baseline_random.csv",
    ("baseline", "scaffold"): "baseline_scaffold.csv",
    ("gnn", "random"): "gnn_random.csv",
    ("gnn", "scaffold"): "gnn_scaffold.csv",
}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_predictions(predictions_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_files: list[Path] = []
    for (_, _), filename in PREDICTION_FILES.items():
        path = predictions_dir / filename
        if not path.exists():
            missing_files.append(path)
            continue
        frames.append(pd.read_csv(path))

    if missing_files:
        paths = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Report generation requires predictions from both models and splits:\n"
            f"{paths}\nRun the baseline and GNN scripts first."
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    """Render a small Markdown table without adding a tabulate dependency."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for record in frame.loc[:, columns].to_dict(orient="records"):
        values: list[str] = []
        for column in columns:
            value = record[column]
            if column in formats and pd.notna(value):
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_report_markdown(
    output_path: Path,
    wide_results: pd.DataFrame,
    failure_cases: pd.DataFrame,
) -> None:
    """Write a compact generated report that avoids causal overinterpretation."""
    result_columns = [
        "model",
        "random_rmse",
        "random_mae",
        "random_r2",
        "scaffold_rmse",
        "scaffold_mae",
        "scaffold_r2",
        "scaffold_minus_random_rmse",
    ]
    number_formats = {column: "{:.3f}" for column in result_columns if column != "model"}
    result_table = _markdown_table(wide_results, result_columns, number_formats)

    scaffold_failures = failure_cases.loc[
        failure_cases["split_method"].astype(str) == "scaffold"
    ].copy()
    failure_columns = [
        "model",
        "smiles",
        "y_true",
        "y_pred",
        "abs_error",
        "nearest_train_tanimoto",
        "chemistry_context",
    ]
    failure_formats = {
        "y_true": "{:.3f}",
        "y_pred": "{:.3f}",
        "abs_error": "{:.3f}",
        "nearest_train_tanimoto": "{:.3f}",
    }
    failure_table = _markdown_table(
        scaffold_failures.head(8), failure_columns, failure_formats
    )

    text = f"""# Generated benchmark report

This file is regenerated from saved prediction artifacts. It does not retrain either model.

## Test-set results

{result_table}

`scaffold_minus_random_rmse` is a descriptive difference, not a significance test.

## Figures

- `figures/parity_random.png`
- `figures/parity_scaffold.png`
- `figures/residuals_random.png`
- `figures/residuals_scaffold.png`
- `figures/applicability_random.png`
- `figures/applicability_scaffold.png`
- `figures/failure_molecules_scaffold.png`

## Largest scaffold-split errors

{failure_table}

The chemistry-context column is descriptor-based and deliberately non-causal. Large errors can reflect representation limits, structural novelty, label noise, assay heterogeneity, solid-state effects, or other factors not resolved by this benchmark.

## Applicability-domain note

`nearest_train_tanimoto` is the maximum radius-2 Morgan fingerprint Tanimoto similarity between a test molecule and the training partition for the same split. It is used only as a structural-proximity diagnostic. This report does not define a universal in-domain threshold and does not treat similarity as calibrated uncertainty.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)


def generate_report(config_path: Path | str, *, root: Path | str = ROOT) -> dict[str, Path]:
    """Generate all report artifacts from saved model outputs."""
    root_path = Path(root)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root_path / config_file
    config = yaml.safe_load(config_file.read_text())

    data_config = config.get("data", {})
    feature_config = config.get("features", {})
    results_config = config.get("results", {})
    report_config = config.get("report", {})

    split_dir = _resolve(root_path, data_config.get("split_dir", "data/processed/splits"))
    tables_dir = _resolve(root_path, results_config.get("tables_dir", "results/tables"))
    figures_dir = _resolve(root_path, results_config.get("figures_dir", "results/figures"))
    predictions_dir = _resolve(
        root_path, results_config.get("predictions_dir", "results/predictions")
    )
    report_path = _resolve(root_path, results_config.get("report_path", "results/report.md"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(predictions_dir)
    radius = int(feature_config.get("morgan_radius", 2))
    n_bits = int(feature_config.get("fingerprint_bits", 2048))

    enriched_frames: list[pd.DataFrame] = []
    for split_method in ("random", "scaffold"):
        split_path = split_dir / f"{split_method}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_frame = pd.read_csv(split_path)
        similarity = nearest_train_tanimoto(
            split_frame,
            radius=radius,
            n_bits=n_bits,
        )
        similarity.to_csv(tables_dir / f"applicability_{split_method}.csv", index=False)

        split_predictions = predictions.loc[
            predictions["split_method"].astype(str) == split_method
        ].copy()
        enriched_frames.append(attach_applicability_domain(split_predictions, similarity))

    enriched = pd.concat(enriched_frames, ignore_index=True, sort=False)
    test_results = benchmark_results(enriched)
    wide_results = wide_benchmark_table(test_results)
    test_results.to_csv(tables_dir / "benchmark_results_long.csv", index=False)
    wide_results.to_csv(tables_dir / "benchmark_summary.csv", index=False)

    applicability = applicability_summary(enriched)
    applicability.to_csv(tables_dir / "applicability_summary.csv", index=False)

    top_n = int(report_config.get("failure_cases_per_model_split", 5))
    failures = select_failure_cases(enriched, top_n_per_model_split=top_n)
    failures.to_csv(tables_dir / "failure_cases.csv", index=False)

    for split_method in ("random", "scaffold"):
        plot_frame = enriched.loc[
            enriched["split_method"].astype(str) == split_method
        ].copy()
        display_split = split_method.capitalize()
        save_parity_plot(
            plot_frame,
            figures_dir / f"parity_{split_method}.png",
            title=f"Test-set parity - {display_split} split",
        )
        save_residual_plot(
            plot_frame,
            figures_dir / f"residuals_{split_method}.png",
            title=f"Test-set residuals - {display_split} split",
        )
        save_applicability_plot(
            plot_frame,
            figures_dir / f"applicability_{split_method}.png",
            title=f"Error vs training-set similarity - {display_split} split",
        )

    save_failure_molecule_grid(
        failures,
        figures_dir / "failure_molecules_scaffold.png",
        split_method="scaffold",
        max_molecules=int(report_config.get("failure_grid_size", 6)),
    )

    history_path = tables_dir / "gnn_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        for split_method in ("random", "scaffold"):
            save_gnn_learning_curve(
                history,
                figures_dir / f"gnn_learning_curve_{split_method}.png",
                split_method=split_method,
            )

    _write_report_markdown(report_path, wide_results, failures)

    return {
        "benchmark_summary": tables_dir / "benchmark_summary.csv",
        "benchmark_results_long": tables_dir / "benchmark_results_long.csv",
        "applicability_summary": tables_dir / "applicability_summary.csv",
        "failure_cases": tables_dir / "failure_cases.csv",
        "report": report_path,
        "parity_scaffold": figures_dir / "parity_scaffold.png",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate benchmark tables and figures from saved predictions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML experiment configuration.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = generate_report(args.config, root=ROOT)
    print("Generated report artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
