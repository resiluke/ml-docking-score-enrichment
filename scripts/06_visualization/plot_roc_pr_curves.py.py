# -*- coding: utf-8 -*-

"""
Generate ROC and Precision-Recall curves for all models.

For each model, the script loads five fold prediction files:
    in_vitro_20_MODEL_fold1.csv
    ...
    in_vitro_20_MODEL_fold5.csv

Each file must contain:
    name
    expected
    predicted

Outputs:
    roc_curves.pdf
    roc_curves_zoom.pdf
    pr_curves.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# plot_roc_pr_curves.py
# ============================================================
# This script generates ROC and Precision-Recall curves for
# all evaluated models.
#
# Input files:
#   in_vitro_20_MODEL_fold*.csv
#
# The script calculates ROC and PR curves for each fold separately,
# interpolates them to a common grid, and then plots the mean curve
# for each model.
#
# The independent in vitro test set is already represented in the
# fold prediction CSV files.
# ============================================================


# ===================== SETTINGS =====================

DATA_DIR = Path(".")
OUT_DIR = Path(".")

N_FOLDS = 5
DS_THRESHOLD = -8.8626

MODELS = {
    "NN1":  {"label": "NN1 (lower bound)", "color": "#888888", "ls": "--"},
    "NN2A": {"label": "NN2A",              "color": "#E377C2", "ls": "-"},
    "NN2B": {"label": "NN2B",              "color": "#2CA02C", "ls": "-"},
    "NN3A": {"label": "NN3A",              "color": "#D62728", "ls": "-"},
    "NN3B": {"label": "NN3B",              "color": "#9467BD", "ls": "-"},
    "NN4A": {"label": "NN4A",              "color": "#FF7F0E", "ls": "-"},
    "NN4B": {"label": "NN4B",              "color": "#1F77B4", "ls": "-"},
    "NN":   {"label": "NN (upper bound)", "color": "#000000", "ls": "--"},
}


# ===================== FUNCTIONS =====================

def load_fold(model_name, fold_idx):
    """Load one prediction CSV file for a selected model and fold."""

    file_path = DATA_DIR / f"in_vitro_20_{model_name}_fold{fold_idx}.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, sep=";")
    df.columns = [col.strip().lower() for col in df.columns]

    return df["expected"].values, df["predicted"].values


def docking_scores_to_binary(expected, threshold=DS_THRESHOLD):
    """
    Convert docking scores to binary labels.
    1 = active/promising compound, DS <= threshold
    0 = inactive/non-promising compound, DS > threshold
    """

    return (expected <= threshold).astype(int)


def compute_roc_folds(model_name):
    """Compute mean ROC curve and AUC across folds."""

    mean_fpr = np.linspace(0, 1, 200)

    all_tpr = []
    all_auc = []

    for fold in range(1, N_FOLDS + 1):

        expected, predicted = load_fold(model_name, fold)

        y_true = docking_scores_to_binary(expected)

        # Lower docking score means better binding.
        # Therefore, the sign is inverted for ROC ranking.
        scores = -predicted

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        all_tpr.append(interp_tpr)
        all_auc.append(roc_auc)

    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": np.std(all_tpr, axis=0),
        "mean_auc": np.mean(all_auc),
        "std_auc": np.std(all_auc, ddof=1),
    }


def compute_pr_folds(model_name):
    """Compute mean Precision-Recall curve and AP across folds."""

    mean_recall = np.linspace(0, 1, 200)

    all_precision = []
    all_ap = []

    for fold in range(1, N_FOLDS + 1):

        expected, predicted = load_fold(model_name, fold)

        y_true = docking_scores_to_binary(expected)
        scores = -predicted

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)

        sorted_idx = np.argsort(recall)
        recall_sorted = recall[sorted_idx]
        precision_sorted = precision[sorted_idx]

        interp_precision = np.interp(
            mean_recall,
            recall_sorted,
            precision_sorted
        )

        all_precision.append(interp_precision)
        all_ap.append(ap)

    return {
        "mean_recall": mean_recall,
        "mean_precision": np.mean(all_precision, axis=0),
        "std_precision": np.std(all_precision, axis=0),
        "mean_ap": np.mean(all_ap),
        "std_ap": np.std(all_ap, ddof=1),
    }


def plot_roc_all_models():
    """Plot mean ROC curves for all models."""

    fig, ax = plt.subplots(figsize=(8, 7))

    curves = []

    for model_name, props in MODELS.items():
        try:
            roc_data = compute_roc_folds(model_name)
            curves.append((roc_data["mean_auc"], props, roc_data))
        except FileNotFoundError as error:
            print(f"[SKIP] {model_name}: missing file {error}")
            continue

    curves.sort(key=lambda item: item[0])

    for _, props, roc_data in curves:
        ax.plot(
            roc_data["mean_fpr"],
            roc_data["mean_tpr"],
            color=props["color"],
            linestyle=props["ls"],
            linewidth=2,
            label=(
                f"{props['label']} "
                f"(AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f})"
            )
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Random classifier"
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False positive rate", fontsize=13)
    ax.set_ylabel("True positive rate", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = OUT_DIR / "roc_curves.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f">> ROC curves saved: {out_path}")


def plot_roc_zoom_all_models():
    """Plot zoomed ROC curves for all models."""

    fig, ax = plt.subplots(figsize=(8, 7))

    curves = []

    for model_name, props in MODELS.items():
        try:
            roc_data = compute_roc_folds(model_name)
            curves.append((roc_data["mean_auc"], props, roc_data))
        except FileNotFoundError as error:
            print(f"[SKIP] {model_name}: missing file {error}")
            continue

    curves.sort(key=lambda item: item[0])

    for _, props, roc_data in curves:
        ax.plot(
            roc_data["mean_fpr"],
            roc_data["mean_tpr"],
            color=props["color"],
            linestyle=props["ls"],
            linewidth=2,
            label=(
                f"{props['label']} "
                f"(AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f})"
            )
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Random classifier"
    )

    ax.set_xlim([0.00, 0.25])
    ax.set_ylim([0.50, 1.02])
    ax.set_xlabel("False positive rate", fontsize=13)
    ax.set_ylabel("True positive rate", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = OUT_DIR / "roc_curves_zoom.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f">> Zoomed ROC curves saved: {out_path}")


def plot_pr_all_models():
    """Plot mean Precision-Recall curves for all models."""

    fig, ax = plt.subplots(figsize=(8, 7))

    curves = []

    for model_name, props in MODELS.items():
        try:
            pr_data = compute_pr_folds(model_name)
            curves.append((pr_data["mean_ap"], props, pr_data))
        except FileNotFoundError as error:
            print(f"[SKIP] {model_name}: missing file {error}")
            continue

    curves.sort(key=lambda item: item[0])

    for _, props, pr_data in curves:
        ax.plot(
            pr_data["mean_recall"],
            pr_data["mean_precision"],
            color=props["color"],
            linestyle=props["ls"],
            linewidth=2,
            label=(
                f"{props['label']} "
                f"(AP = {pr_data['mean_ap']:.3f} ± {pr_data['std_ap']:.3f})"
            )
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.00, 1.02])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = OUT_DIR / "pr_curves.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f">> Precision-Recall curves saved: {out_path}")


# ===================== MAIN =====================

if __name__ == "__main__":

    print("=" * 60)
    print("Generating ROC and Precision-Recall curves")
    print("=" * 60)

    test_file = DATA_DIR / "in_vitro_20_NN1_fold1.csv"

    if not test_file.exists():
        print(f"\nTest file does not exist: {test_file}")
        print("Check DATA_DIR and file naming.")
        print("Expected naming: in_vitro_20_{MODEL}_fold{1-5}.csv")
        print(f"Models: {list(MODELS.keys())}")
    else:
        print(f"\nFiles found in: {DATA_DIR}")

        print("\n[1/3] Generating ROC curves...")
        plot_roc_all_models()

        print("[2/3] Generating zoomed ROC curves...")
        plot_roc_zoom_all_models()

        print("[3/3] Generating Precision-Recall curves...")
        plot_pr_all_models()

        print("\nDone. Generated files:")
        print("  - roc_curves.pdf")
        print("  - roc_curves_zoom.pdf")
        print("  - pr_curves.pdf")