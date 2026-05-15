# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    mean_squared_error, roc_curve, auc,
    precision_recall_curve
)


# ============================================================
# compute_fold_metrics_summary.py
# ============================================================
# This script computes evaluation metrics separately for each
# fold prediction file and then calculates the mean and standard
# deviation across folds.
#
# This approach evaluates the stability of each model across
# five independently trained fold models.
#
# Input files:
#   in_vitro_20_MODEL_fold*.csv
#
# Each input file must contain:
#   name
#   expected
#   predicted
#
# Output files:
#   results_per_fold/all_folds_metrics.csv
#   results_per_fold/summary_mean_std.csv
# ============================================================


# ===================== SETTINGS =====================

DS_THRESHOLD = -8.8626
CSV_SEP = ";"

MODEL_NAMES = [
    "NN1", "NN2A", "NN2B",
    "NN3A", "NN3B",
    "NN4A", "NN4B",
    "NN"
]

CSV_PATTERN_TEMPLATE = "in_vitro_20_{model}_fold*.csv"

OUT_DIR = "results_per_fold"
os.makedirs(OUT_DIR, exist_ok=True)


# ===================== FUNCTIONS =====================

def safe_mse(y_true, y_pred):
    """Return MSE or NaN if the subset is empty."""
    if len(y_true) == 0:
        return np.nan
    return mean_squared_error(y_true, y_pred)


def compute_metrics(y_exp, y_pred, threshold):
    """Compute regression, classification, ROC, PR, and quadrant metrics."""

    y_true_cls = (y_exp <= threshold).astype(int)
    y_pred_cls = (y_pred <= threshold).astype(int)

    # Lower docking score means better activity,
    # therefore the sign is inverted for ROC and PR curves.
    y_scores = -y_pred

    fpr, tpr, _ = roc_curve(y_true_cls, y_scores)
    roc_auc_val = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true_cls,
        y_scores
    )
    pr_auc_val = auc(recall_curve, precision_curve)

    mask_tp = (y_exp <= threshold) & (y_pred <= threshold)
    mask_fp = (y_exp > threshold) & (y_pred <= threshold)
    mask_fn = (y_exp <= threshold) & (y_pred > threshold)
    mask_tn = (y_exp > threshold) & (y_pred > threshold)

    return {
        "ROC_AUC": roc_auc_val,
        "PR_AUC": pr_auc_val,
        "Accuracy": accuracy_score(y_true_cls, y_pred_cls),
        "Balanced_Acc": balanced_accuracy_score(y_true_cls, y_pred_cls),
        "Precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "Recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "F1_Score": f1_score(y_true_cls, y_pred_cls, zero_division=0),
        "MCC": matthews_corrcoef(y_true_cls, y_pred_cls),
        "MSE_Total": mean_squared_error(y_exp, y_pred),
        "MSE_TP": safe_mse(y_exp[mask_tp], y_pred[mask_tp]),
        "MSE_FP": safe_mse(y_exp[mask_fp], y_pred[mask_fp]),
        "MSE_FN": safe_mse(y_exp[mask_fn], y_pred[mask_fn]),
        "MSE_TN": safe_mse(y_exp[mask_tn], y_pred[mask_tn]),
        "TP": int(np.sum(mask_tp)),
        "FP": int(np.sum(mask_fp)),
        "FN": int(np.sum(mask_fn)),
        "TN": int(np.sum(mask_tn)),
    }


# ===================== MAIN LOOP =====================

all_fold_rows = []
summary_rows = []

metric_cols = [
    "ROC_AUC", "PR_AUC",
    "Accuracy", "Balanced_Acc",
    "Precision", "Recall", "F1_Score", "MCC",
    "MSE_Total", "MSE_TP", "MSE_FP", "MSE_FN", "MSE_TN"
]

for model_name in MODEL_NAMES:

    pattern = CSV_PATTERN_TEMPLATE.format(model=model_name)
    csv_files = sorted(glob(pattern))

    if len(csv_files) == 0:
        print(f"[SKIP] No files found for {model_name}")
        continue

    print(f"\n>> Model: {model_name}")
    print(f">> Number of folds: {len(csv_files)}")

    fold_metrics = []

    for fold_id, csv_file in enumerate(csv_files, start=1):

        df = pd.read_csv(csv_file, sep=CSV_SEP).dropna()

        y_exp = df["expected"].values
        y_pred = df["predicted"].values

        metrics = compute_metrics(y_exp, y_pred, DS_THRESHOLD)

        metrics["Model"] = model_name
        metrics["Fold"] = fold_id
        metrics["File"] = os.path.basename(csv_file)

        fold_metrics.append(metrics)
        all_fold_rows.append(metrics)

        print(
            f"   Fold {fold_id}: "
            f"ROC_AUC={metrics['ROC_AUC']:.3f}, "
            f"PR_AUC={metrics['PR_AUC']:.3f}, "
            f"F1={metrics['F1_Score']:.3f}, "
            f"MSE={metrics['MSE_Total']:.3f}"
        )

    df_folds = pd.DataFrame(fold_metrics)

    summary = {
        "Model": model_name,
        "N_Folds": len(csv_files)
    }

    for col in metric_cols:
        values = df_folds[col].dropna().values

        summary[f"{col}_mean"] = np.mean(values)
        summary[f"{col}_std"] = np.std(values, ddof=1)

    summary_rows.append(summary)

    print(
        f"   >>> Mean ± std: "
        f"ROC_AUC={summary['ROC_AUC_mean']:.3f} ± {summary['ROC_AUC_std']:.3f}, "
        f"PR_AUC={summary['PR_AUC_mean']:.3f} ± {summary['PR_AUC_std']:.3f}, "
        f"F1={summary['F1_Score_mean']:.3f} ± {summary['F1_Score_std']:.3f}, "
        f"MSE={summary['MSE_Total_mean']:.3f} ± {summary['MSE_Total_std']:.3f}"
    )


# ===================== SAVE RESULTS =====================

if all_fold_rows:
    df_all = pd.DataFrame(all_fold_rows)

    cols_order = ["Model", "Fold", "File"] + [
        col for col in df_all.columns
        if col not in ["Model", "Fold", "File"]
    ]

    df_all = df_all[cols_order]

    out_file = os.path.join(OUT_DIR, "all_folds_metrics.csv")
    df_all.to_csv(out_file, sep=CSV_SEP, index=False)

    print(f"\n>> Saved fold metrics: {out_file}")


if summary_rows:
    df_summary = pd.DataFrame(summary_rows)

    out_file = os.path.join(OUT_DIR, "summary_mean_std.csv")
    df_summary.to_csv(out_file, sep=CSV_SEP, index=False)

    print(f">> Saved summary metrics: {out_file}")


print("\n>> Done!")