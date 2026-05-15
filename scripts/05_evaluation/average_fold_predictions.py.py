import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error, r2_score

# ===================== SETTINGS =====================
MODEL_NAME = "NNNC"   # change to NN1, NN2, NN3, NN4, NN5
CSV_PATTERN = f"in_vitro_20_{MODEL_NAME}_fold*.csv"

print(f">> Processing {MODEL_NAME}")

# ===================== LOAD CSVs =====================
csv_files = sorted(glob(CSV_PATTERN))
assert len(csv_files) > 0, "No CSV files found!"

dfs = [pd.read_csv(f, sep=";") for f in csv_files]

print(f">> Loaded {len(dfs)} folds")

# ===================== CONSISTENCY CHECK =====================
names = dfs[0]["name"].values
expected = dfs[0]["expected"].values

for df in dfs[1:]:
    assert np.all(df["name"].values == names), "Name mismatch!"
    assert np.all(df["expected"].values == expected), "Expected mismatch!"

# ===================== AVERAGE PREDICTIONS =====================
pred_matrix = np.vstack([df["predicted"].values for df in dfs])
pred_mean = pred_matrix.mean(axis=0)
pred_std  = pred_matrix.std(axis=0)# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# average_fold_predictions.py
# ============================================================
# This script averages prediction results from five fold models.
#
# For a selected model, it loads all CSV files matching:
#   in_vitro_20_MODEL_fold*.csv
#
# Each CSV file must contain:
#   name
#   expected
#   predicted
#
# The script checks that molecule names and expected values are
# identical in all folds. Then it calculates:
#   - mean prediction across folds,
#   - standard deviation of predictions across folds,
#   - MSE based on averaged predictions,
#   - R2 based on averaged predictions.
#
# The output file is:
#   in_vitro_20_MODEL_mean.csv
# ============================================================


# ===================== SETTINGS =====================

MODEL_NAME = "NN"   # examples: NN, NN1, NN2A, NN2B, NN3A, NN3B, NN4A, NN4B
CSV_PATTERN = f"in_vitro_20_{MODEL_NAME}_fold*.csv"

print(f">> Processing model: {MODEL_NAME}")


# ===================== LOAD CSV FILES =====================

csv_files = sorted(glob(CSV_PATTERN))

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found for pattern: {CSV_PATTERN}")

dfs = [pd.read_csv(file, sep=";") for file in csv_files]

print(f">> Loaded {len(dfs)} fold files")


# ===================== CONSISTENCY CHECK =====================

names = dfs[0]["name"].values
expected = dfs[0]["expected"].values

for file, df in zip(csv_files[1:], dfs[1:]):
    if not np.all(df["name"].values == names):
        raise ValueError(f"Name mismatch in file: {file}")

    if not np.allclose(df["expected"].values, expected):
        raise ValueError(f"Expected value mismatch in file: {file}")

print(">> Consistency check passed")


# ===================== AVERAGE FOLD PREDICTIONS =====================

pred_matrix = np.vstack([
    df["predicted"].values
    for df in dfs
])

pred_mean = pred_matrix.mean(axis=0)
pred_std = pred_matrix.std(axis=0)


# ===================== METRICS FROM AVERAGED PREDICTIONS =====================

mse = mean_squared_error(expected, pred_mean)
r2 = r2_score(expected, pred_mean)

print(f"\n>> {MODEL_NAME} results from averaged predictions")
print(f"   MSE: {mse:.3f}")
print(f"   R2:  {r2:.3f}")


# ===================== SAVE MEAN PREDICTION CSV =====================

df_mean = pd.DataFrame({
    "name": names,
    "expected": expected,
    "predicted_mean": pred_mean,
    "predicted_std": pred_std
})

out_name = f"in_vitro_20_{MODEL_NAME}_mean.csv"
df_mean.to_csv(out_name, sep=";", index=False)

print(f">> Saved: {out_name}")

# ===================== METRICS =====================
mse = mean_squared_error(expected, pred_mean)
r2  = r2_score(expected, pred_mean)

print(f">> {MODEL_NAME} MEAN RESULTS")
print(f"   MSE: {mse:.3f}")
print(f"   R2:  {r2:.3f}")

# ===================== SAVE MEAN CSV =====================
df_mean = pd.DataFrame({
    "name": names,
    "expected": expected,
    "predicted_mean": pred_mean,
    "predicted_std": pred_std
})

out_name = f"in_vitro_20_{MODEL_NAME}_mean.csv"
df_mean.to_csv(out_name, sep=";", index=False)

print(f">> Saved: {out_name}")
