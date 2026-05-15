# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# plot_atom_count_distribution.py
# ============================================================
# This script compares molecular size distributions of the
# original in vivo and in vitro datasets.
#
# Input files:
#   in_vivo.xyz
#   in_vitro.xyz
#
# Output file:
#   plots/atom_distribution_comparison.png
# ============================================================


IN_VIVO_XYZ = "in_vivo.xyz"
IN_VITRO_XYZ = "in_vitro.xyz"

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "In-vivo": "#2ca02c",
    "In-vitro": "#9467bd",
}


def extract_atom_counts(filepath):
    """Parse an XYZ file and extract atom counts."""

    atom_counts = []

    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return None

    print(f">> Parsing: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

        while True:
            line = file.readline()

            if not line:
                break

            line = line.strip()

            if not line:
                continue

            try:
                n_atoms = int(line)
            except ValueError:
                continue

            file.readline()

            for _ in range(n_atoms):
                file.readline()

            atom_counts.append(n_atoms)

    atom_counts = np.array(atom_counts)

    print(f"   Loaded: {len(atom_counts):,} compounds")

    return atom_counts


def plot_combined_analysis(datasets, filename):
    """Create a combined histogram for in vivo and in vitro datasets."""

    fig, ax = plt.subplots(figsize=(12, 7))

    all_values = np.concatenate([
        data["counts"]
        for data in datasets.values()
    ])

    bins = np.arange(0, all_values.max() + 5, 2)

    for i, (name, data) in enumerate(datasets.items()):

        counts = data["counts"]
        color = data["color"]

        ax.hist(
            counts,
            bins=bins,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4
        )

        stats_text = (
            f"{name}\n"
            f"n = {len(counts):,}\n"
            f"{counts.min()}–{counts.max()} atoms"
        )

        x_pos = 0.03 if i == 0 else 0.97
        ha = "left" if i == 0 else "right"

        ax.text(
            x_pos,
            0.96,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment=ha,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=color,
                alpha=0.10,
                edgecolor=color,
                linewidth=0.8
            )
        )

    ax.set_xlabel("Number of atoms per molecule", fontsize=12)
    ax.set_ylabel("Number of molecules", fontsize=12)

    ax.grid(axis="y", alpha=0.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f">> Saved: {filename}")


if __name__ == "__main__":

    datasets = {}

    vitro_counts = extract_atom_counts(IN_VITRO_XYZ)

    if vitro_counts is not None:
        datasets["In-vitro"] = {
            "counts": vitro_counts,
            "color": COLORS["In-vitro"]
        }

    vivo_counts = extract_atom_counts(IN_VIVO_XYZ)

    if vivo_counts is not None:
        datasets["In-vivo"] = {
            "counts": vivo_counts,
            "color": COLORS["In-vivo"]
        }

    if len(datasets) == 2:
        plot_combined_analysis(
            datasets,
            os.path.join(
                OUT_DIR,
                "atom_distribution_comparison.png"
            )
        )
    else:
        print("One or more datasets are missing.")

    print("\nDone.")