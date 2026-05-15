# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# plot_docking_score_distribution.py
# ============================================================
# This script analyses the distribution of docking scores in the
# original in vivo and in vitro datasets.
#
# Input files:
#   in_vivo.xyz
#   in_vitro.xyz
#
# The docking scores are extracted from the energy field in the
# XYZ files. The script creates individual histograms for both
# datasets and one combined normalized histogram.
#
# Output files:
#   plots/histogram_in_vivo.png
#   plots/histogram_in_vitro.png
#   plots/histogram_combined.png
# ============================================================


IN_VIVO_XYZ = "in_vivo.xyz"
IN_VITRO_XYZ = "in_vitro.xyz"

DS_THRESHOLD = -8.8626

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "In-vivo": "#2ca02c",
    "In-vitro": "#9467bd",
}


def load_energies(filepath):
    """Extract docking scores from the energy field in an XYZ file."""

    energies = []
    pattern = re.compile(r"energy=([+-]?\d+(?:\.\d+)?)")

    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))

    energies = np.array(energies)

    print(f"{filepath}: {len(energies):,} compounds")

    return energies


def style_axis(ax):
    """Apply a simple publication-style axis format."""

    ax.grid(axis="y", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


def plot_single_histogram(energies, label, color, filename):
    """Create one docking-score histogram for a single dataset."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        energies,
        bins=100,
        color=color,
        alpha=0.70,
        edgecolor="white",
        linewidth=0.4,
        label=f"{label} (n={len(energies):,})"
    )

    ax.axvline(
        DS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Top 10% threshold ({DS_THRESHOLD:.2f})"
    )

    ax.set_xlabel(r"Docking score [kcal$\cdot$mol$^{-1}$]")
    ax.set_ylabel("Number of compounds")

    ax.legend(frameon=True)
    style_axis(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


def plot_combined_histogram(data, filename):
    """Create a combined normalized docking-score histogram."""

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, energies in data.items():

        ax.hist(
            energies,
            bins=100,
            density=True,
            color=COLORS[label],
            alpha=0.50,
            edgecolor="white",
            linewidth=0.3,
            label=f"{label} (n={len(energies):,})"
        )

    ax.axvline(
        DS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Top 10% threshold ({DS_THRESHOLD:.2f})"
    )

    ax.set_xlabel(r"Docking score [kcal$\cdot$mol$^{-1}$]")
    ax.set_ylabel("Probability density")

    ax.legend(frameon=True)
    style_axis(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


if __name__ == "__main__":

    datasets = {}

    vivo = load_energies(IN_VIVO_XYZ)
    if vivo is not None:
        datasets["In-vivo"] = vivo

        plot_single_histogram(
            vivo,
            "In-vivo",
            COLORS["In-vivo"],
            os.path.join(OUT_DIR, "histogram_in_vivo.png")
        )

    vitro = load_energies(IN_VITRO_XYZ)
    if vitro is not None:
        datasets["In-vitro"] = vitro

        plot_single_histogram(
            vitro,
            "In-vitro",
            COLORS["In-vitro"],
            os.path.join(OUT_DIR, "histogram_in_vitro.png")
        )

    if len(datasets) == 2:
        plot_combined_histogram(
            datasets,
            os.path.join(OUT_DIR, "histogram_combined.png")
        )

    print("\nDone.")