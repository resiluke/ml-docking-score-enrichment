# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# plot_summary_metric_bars.py
# ============================================================
# This script generates summary plots for the diploma thesis
# based on the mean ± standard deviation metrics calculated
# across five folds.
#
# Input file:
#   results_per_fold/summary_mean_std.csv
#
# Output files:
#   plots/bar_classification.pdf/png
#   plots/bar_mse_all.pdf/png
#   plots/bar_mse_detail.pdf/png
#   plots/recall_vs_precision.pdf/png
#   plots/timeline.pdf/png
# ============================================================


CSV_FILE = "results_per_fold/summary_mean_std.csv"
CSV_SEP = ";"

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE, sep=CSV_SEP)

# Rename old NNNC label if present
df["Model"] = df["Model"].replace("NNNC", "NN")

MODEL_ORDER = ["NN1", "NN2A", "NN2B", "NN3A", "NN3B", "NN4A", "NN4B", "NN"]

df["sort_key"] = df["Model"].map({
    model: i
    for i, model in enumerate(MODEL_ORDER)
})

df = df.sort_values("sort_key").reset_index(drop=True)

COLORS = {
    "NN1":  "#95a5a6",
    "NN2A": "#e377c2",
    "NN2B": "#2ca02c",
    "NN3A": "#d62728",
    "NN3B": "#9467bd",
    "NN4A": "#ff7f0e",
    "NN4B": "#1f77b4",
    "NN":   "#2ecc71",
}

REF_MODELS = ["NN1", "NN"]

print(f">> Loaded {len(df)} models")
print(
    df[
        [
            "Model",
            "ROC_AUC_mean",
            "PR_AUC_mean",
            "Recall_mean",
            "Precision_mean",
            "MSE_Total_mean",
        ]
    ].to_string(index=False)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_reference_lines(ax, metric):
    """Add reference lines for NN1 and NN."""

    nn1_value = df.loc[df["Model"] == "NN1", f"{metric}_mean"].values[0]
    nn_value = df.loc[df["Model"] == "NN", f"{metric}_mean"].values[0]

    ax.axvline(
        nn1_value,
        color=COLORS["NN1"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        zorder=1
    )

    ax.axvline(
        nn_value,
        color=COLORS["NN"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        zorder=1
    )


def smart_xlim(values, stds, pad_frac=0.15):
    """Calculate compact x-axis limits."""

    lower = (values - stds).min()
    upper = (values + stds).max()
    padding = (upper - lower) * pad_frac

    return max(0, lower - padding), upper + padding


def save_figure(fig, filename_base):
    """Save figure as PDF and PNG."""

    pdf_path = os.path.join(OUT_DIR, f"{filename_base}.pdf")
    png_path = os.path.join(OUT_DIR, f"{filename_base}.png")

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f">> Saved: {filename_base}.pdf / {filename_base}.png")


# ============================================================
# PLOT 1: CLASSIFICATION METRICS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plt.subplots_adjust(hspace=0.35, wspace=0.30)

classification_metrics = [
    ("ROC_AUC", "ROC AUC", axes[0, 0]),
    ("PR_AUC", "PR AUC", axes[0, 1]),
    ("Recall", "Recall", axes[1, 0]),
    ("Precision", "Precision", axes[1, 1]),
]

for metric, title, ax in classification_metrics:

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    colors = [COLORS[model] for model in df["Model"]]

    xmin, xmax = smart_xlim(
        df[mean_col].values,
        df[std_col].values
    )

    bars = ax.barh(
        df["Model"],
        df[mean_col],
        xerr=df[std_col],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        alpha=0.85,
        height=0.65,
        zorder=3
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        value = row[mean_col]
        std = row[std_col]

        x_text = min(
            value + std + (xmax - xmin) * 0.02,
            xmax - (xmax - xmin) * 0.08
        )

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color="#2c3e50"
        )

    add_reference_lines(ax, metric)

    ax.set_xlim(xmin, xmax)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.tick_params(axis="y", labelsize=9)
    ax.invert_yaxis()

fig.suptitle(
    "Classification metrics – model comparison "
    "(mean ± standard deviation, 5 folds)",
    fontsize=14,
    fontweight="bold",
    y=1.01
)

save_figure(fig, "bar_classification")


# ============================================================
# PLOT 2: TOTAL MSE FOR ALL MODELS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

colors = [COLORS[model] for model in df["Model"]]

bars = ax.barh(
    df["Model"],
    df["MSE_Total_mean"],
    xerr=df["MSE_Total_std"],
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    capsize=3,
    alpha=0.85,
    height=0.6,
    zorder=3
)

for bar, (_, row) in zip(bars, df.iterrows()):

    value = row["MSE_Total_mean"]
    std = row["MSE_Total_std"]

    if value > 1:
        ax.text(
            value - 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ± {std:.2f}",
            va="center",
            ha="right",
            fontsize=7,
            color="white",
            fontweight="bold"
        )
    else:
        ax.text(
            value + std + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color="#2c3e50"
        )

add_reference_lines(ax, "MSE_Total")

ax.set_title(
    "Total MSE – all models "
    "(mean ± standard deviation, 5 folds)",
    fontsize=13,
    fontweight="bold",
    pad=10
)

ax.set_xlabel(r"MSE [kcal$^2\cdot$mol$^{-2}$]", fontsize=11)
ax.grid(axis="x", alpha=0.2, zorder=0)
ax.invert_yaxis()

save_figure(fig, "bar_mse_all")


# ============================================================
# PLOT 3: DETAILED MSE WITHOUT NN3A AND NN4A
# ============================================================

df_clean = df[~df["Model"].isin(["NN3A", "NN4A"])].copy()

fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
plt.subplots_adjust(wspace=0.45)

mse_metrics = [
    ("MSE_Total", "Total MSE"),
    ("MSE_TP", "MSE (TP)"),
    ("MSE_FP", "MSE (FP)"),
    ("MSE_TN", "MSE (TN)"),
    ("MSE_FN", "MSE (FN)"),
]

for (metric, title), ax in zip(mse_metrics, axes):

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    colors = [COLORS[model] for model in df_clean["Model"]]

    xmin, xmax = smart_xlim(
        df_clean[mean_col].values,
        df_clean[std_col].values
    )

    bars = ax.barh(
        df_clean["Model"],
        df_clean[mean_col],
        xerr=df_clean[std_col],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        alpha=0.85,
        height=0.55,
        zorder=3
    )

    for bar, (_, row) in zip(bars, df_clean.iterrows()):

        value = row[mean_col]
        std = row[std_col]

        ax.text(
            value + std + (xmax - xmin) * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#2c3e50"
        )

    ax.set_xlim(xmin, xmax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.tick_params(axis="y", labelsize=9)
    ax.invert_yaxis()

fig.suptitle(
    "Detailed MSE comparison by TP, FP, TN and FN "
    "(without NN3A and NN4A)",
    fontsize=13,
    fontweight="bold",
    y=1.05
)

save_figure(fig, "bar_mse_detail")


# ============================================================
# PLOT 4: RECALL VS PRECISION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for _, row in df.iterrows():

    model = row["Model"]
    color = COLORS[model]

    marker = "s" if model in REF_MODELS else "o"
    size = 120 if model in REF_MODELS else 80

    ax.errorbar(
        row["Recall_mean"],
        row["Precision_mean"],
        xerr=row["Recall_std"],
        yerr=row["Precision_std"],
        fmt="none",
        ecolor=color,
        alpha=0.35,
        capsize=3,
        zorder=2
    )

    ax.scatter(
        row["Recall_mean"],
        row["Precision_mean"],
        c=color,
        s=size,
        marker=marker,
        edgecolors="white",
        linewidth=1.5,
        zorder=4
    )

    dx, dy = 0.006, 0.010

    if model == "NN3A":
        dx, dy = -0.025, -0.018
    elif model == "NN2B":
        dx, dy = -0.025, 0.010
    elif model == "NN4A":
        dx, dy = 0.006, -0.018

    ax.annotate(
        model,
        (row["Recall_mean"] + dx, row["Precision_mean"] + dy),
        fontsize=9,
        fontweight="bold",
        color=color,
        zorder=5
    )

ax.annotate(
    "",
    xy=(0.72, 0.84),
    xytext=(0.63, 0.65),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#bdc3c7",
        linewidth=2,
        linestyle="--"
    )
)

ax.text(
    0.685,
    0.745,
    "ideal\ndirection",
    fontsize=8,
    color="#bdc3c7",
    ha="center",
    style="italic",
    rotation=40
)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)

ax.set_title(
    "Recall vs Precision – model comparison",
    fontsize=13,
    fontweight="bold",
    pad=12
)

ax.grid(alpha=0.2)

recall_values = df["Recall_mean"].values
precision_values = df["Precision_mean"].values

ax.set_xlim(recall_values.min() - 0.04, recall_values.max() + 0.04)
ax.set_ylim(precision_values.min() - 0.06, precision_values.max() + 0.06)

save_figure(fig, "recall_vs_precision")


# ============================================================
# PLOT 5: MODEL PROGRESSION TIMELINE
# ============================================================

higher_better_metrics = [
    "ROC_AUC_mean",
    "PR_AUC_mean",
    "Recall_mean",
    "Precision_mean"
]

lower_better_metrics = [
    "MSE_Total_mean"
]

scores = pd.DataFrame()
scores["Model"] = df["Model"]

for col in higher_better_metrics:

    values = df[col].values
    vmin, vmax = values.min(), values.max()

    if vmax > vmin:
        scores[col] = (values - vmin) / (vmax - vmin)
    else:
        scores[col] = 0.5

for col in lower_better_metrics:

    values = df[col].values
    vmin, vmax = values.min(), values.max()

    if vmax > vmin:
        scores[col] = 1.0 - (values - vmin) / (vmax - vmin)
    else:
        scores[col] = 0.5

df["composite"] = scores[
    higher_better_metrics + lower_better_metrics
].mean(axis=1)

nn1_score = df.loc[df["Model"] == "NN1", "composite"].values[0]
nn_score = df.loc[df["Model"] == "NN", "composite"].values[0]

df["position"] = (
    (df["composite"] - nn1_score) /
    (nn_score - nn1_score)
).clip(-0.05, 1.05)

df_timeline = df.sort_values("position").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15, 6.5))

x_margin = 0.06
x_start = x_margin
x_end = 1.0 - x_margin
y_line = 0.45

ax.annotate(
    "",
    xy=(x_end + 0.03, y_line),
    xytext=(x_start - 0.02, y_line),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#2c3e50",
        linewidth=2.5
    )
)

ax.text(
    x_start - 0.01,
    y_line - 0.07,
    "NN1\n(lower bound)",
    ha="center",
    va="top",
    fontsize=9,
    color=COLORS["NN1"],
    fontweight="bold"
)

ax.text(
    x_end + 0.02,
    y_line - 0.07,
    "NN\n(upper bound)",
    ha="center",
    va="top",
    fontsize=9,
    color=COLORS["NN"],
    fontweight="bold"
)

for idx, (_, row) in enumerate(df_timeline.iterrows()):

    model = row["Model"]
    color = COLORS[model]

    x = x_start + row["position"] * (x_end - x_start)

    above = idx % 2 == 0
    y_text = y_line + (0.20 if above else -0.20)
    y_conn = y_line + (0.10 if above else -0.10)

    dot_size = 16 if model in REF_MODELS else 12

    ax.plot(
        x,
        y_line,
        "o",
        color=color,
        markersize=dot_size,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=5
    )

    ax.plot(
        [x, x],
        [y_line, y_conn],
        color=color,
        linewidth=1.2,
        alpha=0.4,
        zorder=3
    )

    fontweight = "bold" if model in REF_MODELS else "normal"
    fontsize = 12 if model in REF_MODELS else 10

    ax.text(
        x,
        y_text,
        model,
        ha="center",
        va="bottom" if above else "top",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color
    )

    metrics_text = (
        f"ROC: {row['ROC_AUC_mean']:.3f}\n"
        f"PR:  {row['PR_AUC_mean']:.3f}\n"
        f"Rec: {row['Recall_mean']:.3f}\n"
        f"MSE: {row['MSE_Total_mean']:.3f}"
    )

    bbox = dict(
        boxstyle="round,pad=0.25",
        facecolor=color,
        alpha=0.08,
        edgecolor=color,
        linewidth=0.8
    )

    metrics_y = y_text + (0.05 if above else -0.05)

    ax.text(
        x,
        metrics_y,
        metrics_text,
        ha="center",
        va="bottom" if above else "top",
        fontsize=6.5,
        color="#2c3e50",
        bbox=bbox,
        family="monospace"
    )

ax.set_title(
    "Model progression from lower to upper accuracy bound",
    fontsize=14,
    fontweight="bold",
    pad=15,
    color="#2c3e50"
)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.0, 1.0)
ax.axis("off")

save_figure(fig, "timeline")


# ============================================================
# FINAL MESSAGE
# ============================================================

print(f"\n>> All plots saved in: {OUT_DIR}/")
print("Files:")

for file_name in sorted(os.listdir(OUT_DIR)):
    print(f"   {file_name}")