"""Post-process evaluation results: tables, Pareto plots, LaTeX export.

Usage:
    cd examples/eval_real_world
    python -m scripts.analyze_results --results results/results.csv --output_dir results

Outputs:
    pareto_{dataset}.png     – RMSE vs Train Time / Peak Memory
    ablation_bar.png          – Ablation comparison per dataset
    stability_table.tex      – LaTeX table for the paper
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_pareto_fronts(rows: list[dict], output_dir: Path) -> None:
    """Generate Pareto plots: RMSE vs Train Time and RMSE vs Peak Memory."""
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, 12))

    for dataset in datasets:
        fig, axes = plt.subplots(1, len(tiers), figsize=(5 * len(tiers), 5))
        if len(tiers) == 1:
            axes = [axes]

        for ax, tier in zip(axes, tiers, strict=False):
            subset = [r for r in rows if r["dataset"] == dataset and r["tier"] == tier]
            models = sorted({r["model"] for r in subset})
            for idx, model in enumerate(models):
                model_rows = [r for r in subset if r["model"] == model]
                # Parse mean ± std strings
                rmse_parts = model_rows[0]["rmse"].split("±")
                time_parts = model_rows[0]["train_time_sec"].split("±")
                rmse_m = float(rmse_parts[0].strip())
                time_m = float(time_parts[0].strip())
                ax.scatter(time_m, rmse_m, label=model, color=colors[idx % 12], s=80)
                ax.annotate(model, (time_m, rmse_m), fontsize=6, alpha=0.7)

            ax.set_xlabel("Train Time (s)")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{dataset} — {tier}")
            ax.legend(fontsize=5, loc="best")

        fig.tight_layout()
        out_path = output_dir / f"pareto_{dataset}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_ablation_bars(rows: list[dict], output_dir: Path) -> None:
    """Bar plot of RMSE for each ablation relative to full AwareKernel."""
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})
    ablation_names = [
        "AK-NoRefresh",
        "AK-NoHysteresis",
        "AK-NoCooldown",
        "AK-NoResidAnchors",
        "AK-NoOrthog",
        "AK-NoDivPenalty",
        "AK-StaticScaling",
    ]

    for dataset in datasets:
        for tier in tiers:
            subset = [r for r in rows if r["dataset"] == dataset and r["tier"] == tier]
            full = next((r for r in subset if r["model"] == "AwareKernel"), None)
            if full is None:
                continue
            full_rmse = float(full["rmse"].split("±")[0].strip())

            names = []
            rel_rmses = []
            for ab in ablation_names:
                row = next((r for r in subset if r["model"] == ab), None)
                if row:
                    names.append(ab.replace("AK-", ""))
                    r = float(row["rmse"].split("±")[0].strip())
                    rel_rmses.append((r - full_rmse) / full_rmse * 100)

            if not names:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                range(len(names)),
                rel_rmses,
                color=["red" if v > 0 else "green" for v in rel_rmses],
            )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Rel. RMSE change (%)")
            ax.set_title(f"Ablations: {dataset} — {tier}")
            fig.tight_layout()
            out_path = output_dir / f"ablations_{dataset}_{tier}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


def export_latex_table(rows: list[dict], output_dir: Path) -> None:
    """Export a LaTeX table of main results ( AwareKernel + baselines only, no ablations)."""
    datasets = sorted({r["dataset"] for r in rows})
    tiers = sorted({r["tier"] for r in rows})
    models = ["AwareKernel", "Ridge", "Nystrom", "RFF"]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Dataset & Tier & Model & RMSE & Train (s) & Peak Mem (MB) \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        for tier in tiers:
            for model in models:
                row = next(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["tier"] == tier
                        and r["model"] == model
                    ),
                    None,
                )
                if row:
                    lines.append(
                        f"{dataset} & {tier} & {model} & "
                        f"{row['rmse']} & {row['train_time_sec']} & {row['peak_mem_mb']} \\\\"
                    )
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Evaluation results across datasets and budget tiers.}")
    lines.append("\\end{table}")

    out_path = output_dir / "results_table.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation results.")
    parser.add_argument(
        "--results", type=str, required=True, help="Path to results CSV."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for plots."
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_csv(results_path)
    print(f"Loaded {len(rows)} result rows from {results_path}")

    plot_pareto_fronts(rows, output_dir)
    plot_ablation_bars(rows, output_dir)
    export_latex_table(rows, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
