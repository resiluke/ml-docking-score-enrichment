import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ============================================================
# DS_threshold_analysis.py
# ============================================================
# This script analyses the in vitro dataset based on docking scores.
#
# The input file contains molecule names, expected docking scores,
# and predicted docking scores. The script determines the docking
# score threshold from the expected values by selecting the best
# 10% of molecules, i.e. the molecules with the lowest docking scores.
#
# Based on this threshold, the script identifies:
#   - molecules belonging to the top 10% according to expected values,
#   - molecules belonging to the top 10% according to predicted values,
#   - the overlap between both groups.
#
# This allows the evaluation of how well the model predicts the most
# promising compounds from the in vitro dataset.
# ============================================================


# === Parameters ===
name_file = "xyz_energy_export.csv"
percent = 0.10  # top 10% of molecules with the lowest docking scores

# === Load CSV file ===
df = pd.read_csv(name_file, sep=";")

# === Automatic calculation of DS threshold ===
DS_threshold = df["expected"].quantile(percent)

print(
    f"Automatically selected DS_threshold for the best "
    f"{percent*100:.0f}% of molecules = {DS_threshold:.4f} kcal/mol"
)

# === Selection based on the calculated threshold ===
exp_mask = df["expected"] <= DS_threshold
pred_mask = df["predicted"] <= DS_threshold

# === Sets of molecule names for comparison ===
expected_top_names = set(df.loc[exp_mask, "name"])
predicted_top_names = set(df.loc[pred_mask, "name"])

# === Basic counts ===
n_exp = len(expected_top_names)          # positives according to expected DS
n_pred = len(predicted_top_names)        # positives according to predicted DS

intersection = expected_top_names & predicted_top_names
common_count = len(intersection)

percentage = (common_count / n_exp * 100) if n_exp > 0 else np.nan

print(f"DS threshold: {DS_threshold}")
