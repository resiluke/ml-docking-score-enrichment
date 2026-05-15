# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# plot_quadrant_density.py
# ============================================================
# This script generates quadrant density plots for all models.
#
# The quadrant counts TP, FP, FN, and TN are calculated separately
# for each fold and then averaged across folds. These averaged
# counts are used only for plot legends.
#
# The visualization itself uses averaged predictions across folds
# for each molecule.
#
# Input files:
#   in_vitro_20_MODEL_fold1.csv
#   ...
#   in_vitro_20_MODEL_fold5.csv
#
# Output files:
#   figures/MODEL_quadrant_density.png
# ============================================================


DATA_DIR = Path(".")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
DS_THRESHOLD = -8.8626

AXIS_MIN = -14
AXIS_MAX = 0
DPI = 400

MODELS = {
    "NN1":  "NN1 (lower bound)",
    "NN2A": "NN2A",
    "NN2B": "NN2B",
    "NN3A": "NN3A",
    "NN3B": "NN3B",
    "NN4A": "NN4A",
    "NN4B": "NN4B",
    "NN":   "NN (upper bound)",
}

Q_COLORS = {
    "TP": "#1b9e77",
    "FP": "#d95f02",
    "FN": "#e7298a",
    "TN": "#7570b3",
}


def load_fold(model_name, fold_idx):
    """Load one prediction CSV file for a selected model and fold."""

    file_path = DATA_DIR / f"in_vitro_20_{model_name}_fold{fold_idx}.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, sep=";")
    df.columns = [col.strip().lower() for col in df.columns]

    return df


def count_quadrants_per_fold(model_name, threshold):
    """Calculate TP, FP, FN, and TN counts for each fold."""

    fold_counts = []

    for fold in range(1, N_FOLDS + 1):

        df = load_fold(model_name, fold)

        y_exp = df["expected"].values
        y_pred = df["predicted"].values

        tp = np.sum((y_exp <= threshold) & (y_pred <= threshold))
        fp = np.sum((y_exp > threshold) & (y_pred <= threshold))
        fn = np.sum((y_exp <= threshold) & (y_pred > threshold))
        tn = np.sum((y_exp > threshold) & (y_pred > threshold))

        fold_counts.append({
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

    df_counts = pd.DataFrame(fold_counts)

    return df_counts.mean(), df_counts.std(ddof=1), df_counts


def load_averaged_predictions(model_name):
    """Average predictions across folds for each molecule."""

    dfs = []

    for fold in range(1, N_FOLDS + 1):
        dfs.append(load_fold(model_name, fold))

    combined = pd.concat(dfs, ignore_index=True)

    averaged = combined.groupby("name").agg(
        expected=("expected", "mean"),
        predicted=("predicted", "mean")
    ).reset_index()

    return averaged["expected"].values, averaged["predicted"].values


def density_2d(x, y, bins=250):
    """Calculate local 2D point density."""

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    x_idx = np.clip(
        np.searchsorted(x_edges, x, side="right") - 1,
        0,
        bins - 1
    )

    y_idx = np.clip(
        np.searchsorted(y_edges, y, side="right") - 1,
        0,
        bins - 1
    )

    return hist[x_idx, y_idx]


def plot_density_quadrants(
    y_exp,
    y_pred,
    model_name,
    display_name,
    threshold,
    quadrant_means
):
    """Create a density scatter plot divided into TP, FP, FN, and TN quadrants."""

    mask_range = (
        (y_exp >= AXIS_MIN) &
        (y_exp <= AXIS_MAX) &
        (y_pred >= AXIS_MIN) &
        (y_pred <= AXIS_MAX)
    )

    y_exp = y_exp[mask_range]
    y_pred = y_pred[mask_range]

    density = density_2d(y_exp, y_pred)

    norm = Normalize(
        vmin=np.percentile(density, 5),
        vmax=np.percentile(density, 95)
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    masks = {
        "TP": (y_exp <= threshold) & (y_pred <= threshold),
        "FP": (y_exp > threshold) & (y_pred <= threshold),
        "FN": (y_exp <= threshold) & (y_pred > threshold),
        "TN": (y_exp > threshold) & (y_pred > threshold),
    }

    legend_handles = []

    for quadrant, mask in masks.items():

        if not np.any(mask):
            continue

        x_q = y_exp[mask]
        y_q = y_pred[mask]
        d_q = density[mask]

        order = np.argsort(d_q)

        x_q = x_q[order]
        y_q = y_q[order]
        d_q = d_q[order]

        cmap = LinearSegmentedColormap.from_list(
            f"cmap_{quadrant}",
            ["#f2f2f2", Q_COLORS[quadrant]]
        )

        ax.scatter(
            x_q,
            y_q,
            c=d_q,
            cmap=cmap,
            norm=norm,
            s=14,
            edgecolors=Q_COLORS[quadrant],
            linewidths=0.4,
            rasterized=True
        )

        n_average = round(quadrant_means[quadrant])

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{quadrant} (N={n_average})",
                markerfacecolor=Q_COLORS[quadrant],
                markeredgecolor=Q_COLORS[quadrant],
                markersize=8
            )
        )

    ax.axvline(
        threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.5
    )

    ax.axhline(
        threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.5
    )

    ax.plot(
        [AXIS_MIN, AXIS_MAX],
        [AXIS_MIN, AXIS_MAX],
        color="#bbbbbb",
        linestyle=":",
        linewidth=1
    )

    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)

    ax.set_xlabel("Experimental DS [kcal/mol]", fontsize=11)
    ax.set_ylabel("Predicted DS [kcal/mol]", fontsize=11)

    ax.set_title(display_name, fontsize=13, fontweight="bold", pad=10)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=11
    )

    ax.text(
        threshold + 0.2,
        AXIS_MIN + 0.5,
        f"Threshold: {threshold:.4f}",
        fontsize=9,
        fontweight="bold"
    )

    plt.tight_layout()

    output_file = OUT_DIR / f"{model_name}_quadrant_density.png"

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"   Saved: {output_file}")


if __name__ == "__main__":

    print("=" * 60)
    print("Generating quadrant density plots")
    print("=" * 60)

    test_file = DATA_DIR / "in_vitro_20_NN1_fold1.csv"

    if not test_file.exists():
        print(f"\nTest file does not exist: {test_file}")
        print("Check DATA_DIR and file naming.")
        print("Expected naming: in_vitro_20_{MODEL}_fold{1-5}.csv")
        print(f"Models: {list(MODELS.keys())}")

    else:
        print(f"\nFiles found in: {DATA_DIR}\n")

        print(f"{'Model':<18} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>10}")
        print("-" * 60)

        for model_name, display_name in MODELS.items():

            try:
                means, stds, fold_counts = count_quadrants_per_fold(
                    model_name,
                    DS_THRESHOLD
                )

                print(
                    f"{display_name:<18} "
                    f"{round(means['TP']):>8} "
                    f"{round(means['FP']):>8} "
                    f"{round(means['FN']):>8} "
                    f"{round(means['TN']):>10}"
                )

                y_exp, y_pred = load_averaged_predictions(model_name)

                plot_density_quadrants(
                    y_exp=y_exp,
                    y_pred=y_pred,
                    model_name=model_name,
                    display_name=display_name,
                    threshold=DS_THRESHOLD,
                    quadrant_means=means
                )

            except FileNotFoundError as error:
                print(f"[SKIP] {model_name}: missing file {error}")

            except Exception as error:
                print(f"[ERROR] {model_name}: {error}")

        print("-" * 60)
        print(f"\nDone. Outputs saved in: {OUT_DIR.resolve()}")