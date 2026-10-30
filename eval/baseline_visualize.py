
"""Visualize local-only, pooled, and FedAvg baseline results."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "local_only": "Local-only",
    "pooled": "Pooled",
    "fedavg": "FedAvg",
}

CENTER_LABELS = {
    0: "Cleveland",
    1: "Hungary",
    2: "Switzerland",
    3: "Long Beach VA",
}

METHOD_ORDER = ["local_only", "pooled", "fedavg"]


def visualize_baselines(csv_path, output_dir="results/figures"):
    """Generate AUROC, sensitivity, and overall comparison charts.

    CSV columns required:
        method, seed, center, auroc, sensitivity

    Error bars represent one standard deviation across seeds.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"method", "seed", "center", "auroc", "sensitivity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Convert metric columns to numeric and ignore missing measurements.
    for col in ["center", "seed", "auroc", "sensitivity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["method"].isin(METHOD_ORDER)].copy()

    if df.empty:
        raise ValueError("CSV contains no recognized baseline methods.")

    centers = sorted(df["center"].dropna().unique())
    center_labels = [
        CENTER_LABELS.get(int(c), f"Center {int(c)}")
        for c in centers
    ]

    # Calculate mean and standard deviation across seeds per center.
    by_center = (
        df.groupby(["center", "method"])[["auroc", "sensitivity"]]
        .agg(["mean", "std"])
    )

    # Print per-center results.
    print("\nPER-CENTER RESULTS (mean ± standard deviation across seeds)")
    print("=" * 75)

    for center in centers:
        center_name = CENTER_LABELS.get(int(center), f"Center {int(center)}")
        print(f"\n{center_name}")
        print("-" * 75)
        print(f"{'Method':<15} {'AUROC':<24} {'Sensitivity':<24}")

        for method in METHOD_ORDER:
            key = (center, method)

            if key not in by_center.index:
                continue

            auroc_mean = by_center.loc[key, ("auroc", "mean")]
            auroc_std = by_center.loc[key, ("auroc", "std")]
            sens_mean = by_center.loc[key, ("sensitivity", "mean")]
            sens_std = by_center.loc[key, ("sensitivity", "std")]

            print(
                f"{METHOD_LABELS[method]:<15} "
                f"{auroc_mean:.4f} ± {auroc_std:.4f}       "
                f"{sens_mean:.4f} ± {sens_std:.4f}"
            )

    def plot_by_center(metric, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(10, 5.8))
        x = list(range(len(centers)))
        width = 0.24
        offsets = [-width, 0, width]

        for method, offset in zip(METHOD_ORDER, offsets):
            means = []
            stds = []

            for center in centers:
                key = (center, method)
                if key in by_center.index:
                    means.append(by_center.loc[key, (metric, "mean")])
                    stds.append(by_center.loc[key, (metric, "std")])
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))

            # A single seed has no sample standard deviation.
            stds = [0 if pd.isna(s) else s for s in stds]

            ax.bar(
                [v + offset for v in x],
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=METHOD_LABELS[method],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(center_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        path = output_dir / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # Figure 1: AUROC by center.
    plot_by_center(
        metric="auroc",
        title="Baseline AUROC by Center Across Seeds",
        ylabel="AUROC (mean ± standard deviation)",
        filename="baseline_auroc_by_center.png",
    )

    # Figure 2: Sensitivity by center.
    plot_by_center(
        metric="sensitivity",
        title="Baseline Sensitivity by Center Across Seeds",
        ylabel="Sensitivity (mean ± standard deviation)",
        filename="baseline_sensitivity_by_center.png",
    )

    # Figure 3: Overall mean across center-seed observations.
    # Each center-seed observation receives equal weight.
    overall = (
        df.groupby("method")[["auroc", "sensitivity"]]
        .agg(["mean", "std"])
        .reindex(METHOD_ORDER)
    )

    # Print overall results.
    print("\nOVERALL RESULTS (mean ± standard deviation across center-seed observations)")
    print("=" * 75)
    print(f"{'Method':<15} {'AUROC':<24} {'Sensitivity':<24}")

    for method in METHOD_ORDER:
        auroc_mean = overall.loc[method, ("auroc", "mean")]
        auroc_std = overall.loc[method, ("auroc", "std")]
        sens_mean = overall.loc[method, ("sensitivity", "mean")]
        sens_std = overall.loc[method, ("sensitivity", "std")]

        print(
            f"{METHOD_LABELS[method]:<15} "
            f"{auroc_mean:.4f} ± {auroc_std:.4f}       "
            f"{sens_mean:.4f} ± {sens_std:.4f}"
        )

    # Plot overall results.
    fig, ax = plt.subplots(figsize=(8, 5.3))
    x = list(range(len(METHOD_ORDER)))
    width = 0.34

    for metric, offset, label in [
        ("auroc", -width / 2, "AUROC"),
        ("sensitivity", width / 2, "Sensitivity"),
    ]:
        means = overall[(metric, "mean")].tolist()
        stds = overall[(metric, "std")].fillna(0).tolist()

        ax.bar(
            [v + offset for v in x],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Metric (mean ± standard deviation)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Baseline Performance Across Centers and Seeds")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = output_dir / "baseline_overall_metrics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot federated learning baseline metrics."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the baseline metrics CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    visualize_baselines(args.csv_path, args.output_dir)
