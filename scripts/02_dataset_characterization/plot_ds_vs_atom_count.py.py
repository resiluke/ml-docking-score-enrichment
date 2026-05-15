# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# plot_ds_vs_atom_count.py
# ============================================================
# This script analyses the relationship between molecular size
# and docking score in the original in vivo and in vitro datasets.
#
# Input files:
#   in_vivo.xyz
#   in_vitro.xyz
#
# For each molecule, the script extracts:
#   - number of atoms
#   - docking score from the energy field
#
# Output files:
#   plots/ds_vs_atoms_density_in_vivo.png
#   plots/ds_vs_atoms_density_in_vitro.png
#   plots/ds_vs_atoms_mean_in_vivo.png
#   plots/ds_vs_atoms_mean_in_vitro.png
# ============================================================


IN_VIVO_XYZ = "in_vivo.xyz"
IN_VITRO_XYZ = "in_vitro.xyz"

DS_THRESHOLD = -8.8626

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_atoms_and_energies(filepath):
    """Parse XYZ file and return atom counts and docking scores."""

    atom_counts = []
    energies = []

    energy_pattern = re.compile(r"energy=([+-]?\d+(?:\.\d+)?)")

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

            header = file.readline().strip()
            match = energy_pattern.search(header)

            energy = float(match.group(1)) if match else np.nan

            for _ in range(n_atoms):
                file.readline()

            atom_counts.append(n_atoms)
            energies.append(energy)

    atom_counts = np.array(atom_counts)
    energies = np.array(energies)

    valid_mask = ~np.isnan(energies)

    atom_counts = atom_counts[valid_mask]
    energies = energies[valid_mask]

    print(f">> Loaded {len(atom_counts):,} compounds from {filepath}")

    return atom_counts, energies


def point_density(x, y, bins=250):
    """Calculate local 2D point density."""

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    x_idx = np.searchsorted(x_edges, x, side="right") - 1
    y_idx = np.searchsorted(y_edges, y, side="right") - 1

    x_idx = np.clip(x_idx, 0, bins - 1)
    y_idx = np.clip(y_idx, 0, bins - 1)

    return hist[x_idx, y_idx]


def plot_density_scatter(atom_counts, energies, label, filename):
    """Plot docking score versus atom count using local point density."""

    fig, ax = plt.subplots(figsize=(11, 7))

    density = point_density(atom_counts, energies, bins=250)

    if np.any(density > 0):
        p5, p99 = np.percentile(density[density > 0], [5, 99])
    else:
        p5, p99 = 0, 1

    norm = mcolors.Normalize(vmin=p5, vmax=p99, clip=True)

    scatter = ax.scatter(
        atom_counts,
        energies,
        c=density,
        norm=norm,
        cmap="inferno",
        s=12,
        marker="o",
        edgecolors="none",
        alpha=1.0
    )

    ax.axhline(
        y=DS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Top 10% threshold ({DS_THRESHOLD:.2f} kcal/mol)"
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Local density", fontsize=11)

    ax.set_xlabel("Number of atoms per molecule", fontsize=13)
    ax.set_ylabel(r"Docking score [kcal$\cdot$mol$^{-1}$]", fontsize=13)
    ax.set_title(
        f"Relationship between docking score and molecular size – {label}",
        fontsize=14,
        pad=12
    )

    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f">> Saved: {filename}")


def plot_mean_ds_vs_atoms(atom_counts, energies, label, filename):
    """Plot mean docking score as a function of atom count."""

    fig, ax = plt.subplots(figsize=(10, 6))

    bin_edges = np.arange(atom_counts.min(), atom_counts.max() + 5, 5)

    bin_centers = []
    mean_ds = []
    std_ds = []

    for i in range(len(bin_edges) - 1):

        mask = (
            (atom_counts >= bin_edges[i]) &
            (atom_counts < bin_edges[i + 1])
        )

        if np.sum(mask) >= 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            mean_ds.append(energies[mask].mean())
            std_ds.append(energies[mask].std())

    bin_centers = np.array(bin_centers)
    mean_ds = np.array(mean_ds)
    std_ds = np.array(std_ds)

    ax.errorbar(
        bin_centers,
        mean_ds,
        yerr=std_ds,
        fmt="o-",
        color="#1f77b4",
        capsize=3,
        markersize=5,
        linewidth=1.5,
        alpha=0.85,
        label=label
    )

    ax.fill_between(
        bin_centers,
        mean_ds - std_ds,
        mean_ds + std_ds,
        alpha=0.15,
        color="#1f77b4"
    )

    ax.axhline(
        y=DS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Top 10% threshold ({DS_THRESHOLD:.2f} kcal/mol)"
    )

    ax.set_xlabel("Number of atoms per molecule", fontsize=13)
    ax.set_ylabel(r"Mean docking score [kcal$\cdot$mol$^{-1}$]", fontsize=13)
    ax.set_title(
        f"Mean docking score as a function of molecular size – {label}",
        fontsize=14,
        pad=12
    )

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f">> Saved: {filename}")


if __name__ == "__main__":

    datasets = [
        (IN_VIVO_XYZ, "In vivo", "in_vivo"),
        (IN_VITRO_XYZ, "In vitro", "in_vitro"),
    ]

    for xyz_file, label, suffix in datasets:

        if not os.path.exists(xyz_file):
            print(f"[SKIP] File not found: {xyz_file}")
            continue

        print(f"\n>> Processing {label} dataset")

        atom_counts, energies = extract_atoms_and_energies(xyz_file)

        plot_density_scatter(
            atom_counts,
            energies,
            label,
            os.path.join(OUT_DIR, f"ds_vs_atoms_density_{suffix}.png")
        )

        plot_mean_ds_vs_atoms(
            atom_counts,
            energies,
            label,
            os.path.join(OUT_DIR, f"ds_vs_atoms_mean_{suffix}.png")
        )

    print("\nDone.")