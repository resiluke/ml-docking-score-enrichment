# -*- coding: utf-8 -*-

from ase.io import read, write
import tensorflow as tf
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold


# ============================================================
# NN4B_interval_enrichment_5fold.py
# ============================================================
# This script trains the NN4B interval enrichment model using
# five-fold cross-validation.
#
# First, the pre-trained NN1 fold models are used as an ensemble
# to predict docking scores for:
#   in_vitro_80_train.xyz
#
# The 10% docking-score threshold is calculated from the mean
# ensemble predictions. Then, compounds are selected from the
# interval:
#   threshold ± DELTA_DS
#
# If the number of interval candidates is larger than TOP_N,
# a random subset of TOP_N compounds is selected.
#
# The selected interval subset is divided into five folds:
#   interval_80_XX_train.xyz
#   interval_20_XX_test.xyz
#
# For each fold:
#   Training set   = in vivo training + interval training subset
#   Validation set = in vivo validation + interval validation subset
#
# The independent final in vitro test set:
#   in_vitro_20_test.xyz
#
# is kept untouched during selection, training, and validation.
# It is used only for the final evaluation of each trained fold model.
#
# Input and output file names are kept unchanged.
# ============================================================


TOP_N = 29942
DELTA_DS = 1.5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def get_energies(structures):
    """Extract docking scores / potential energies from structures."""
    return np.array([s.get_potential_energy() for s in structures])


def get_names(structures):
    """Extract molecule names from structures."""
    return [
        s.info.get("name", f"mol_{i}")
        for i, s in enumerate(structures)
    ]


def build_model(input_dim):
    """Create the NN4B neural network architecture."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(100, activation="softplus"),
        tf.keras.layers.Dense(50, activation="softplus"),
        tf.keras.layers.Dense(5, activation="softplus"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error"
    )

    return model


print(">> NN4B interval enrichment with five-fold split")


# ============================================================
# 1. LOAD DATA
# ============================================================

vitro_pool = read("in_vitro_80_train.xyz", index=":")
structures_final = read("in_vitro_20_test.xyz", index=":")

y_final = get_energies(structures_final)
final_names = get_names(structures_final)


# ============================================================
# 2. SOAP SPECIES
# ============================================================

all_structures = vitro_pool + structures_final

for i in range(1, 6):
    all_structures += read(f"in_vivo_80_{i:02d}_train.xyz", index=":")
    all_structures += read(f"in_vivo_20_{i:02d}_test.xyz", index=":")

species = sorted(set(
    atom
    for structure in all_structures
    for atom in structure.get_chemical_symbols()
))

soap = SOAP(
    species=species,
    periodic=False,
    r_cut=16.0,
    n_max=5,
    l_max=5,
    sigma=0.5,
    average="outer",
    sparse=False
)


# ============================================================
# 3. NN1 ENSEMBLE PREDICTION
# ============================================================

print(">> Running NN1 ensemble prediction")

X_pool = np.asarray(soap.create(vitro_pool, n_jobs=-1))

ensemble_preds = []

for fold in range(1, 6):
    model = tf.keras.models.load_model(f"NN1_fold{fold}_best.keras")
    preds = model.predict(X_pool, verbose=0).reshape(-1)
    ensemble_preds.append(preds)

mean_preds = np.mean(np.array(ensemble_preds), axis=0)


# ============================================================
# 4. INTERVAL SELECTION
# ============================================================

ds_threshold = np.percentile(mean_preds, 10)

print(f">> 10% DS threshold: {ds_threshold:.3f}")

interval_mask = (
    (mean_preds >= ds_threshold - DELTA_DS) &
    (mean_preds <= ds_threshold + DELTA_DS)
)

interval_indices = np.where(interval_mask)[0]

print(f">> Interval candidates: {len(interval_indices)}")

if len(interval_indices) > TOP_N:
    selected_indices = np.random.choice(
        interval_indices,
        size=TOP_N,
        replace=False
    )
else:
    selected_indices = interval_indices
    print(">> Interval smaller than TOP_N, using all candidates")

interval_structures = [vitro_pool[i] for i in selected_indices]

print(f">> Final interval size: {len(interval_structures)}")


# ============================================================
# 5. CREATE FIVE CROSS-VALIDATION FOLDS FOR INTERVAL DATA
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

for fold_id, (train_idx, test_idx) in enumerate(kf.split(interval_structures), start=1):

    interval_train = [interval_structures[i] for i in train_idx]
    interval_test = [interval_structures[i] for i in test_idx]

    write(f"interval_80_{fold_id:02d}_train.xyz", interval_train)
    write(f"interval_20_{fold_id:02d}_test.xyz", interval_test)

print(">> Interval 5-fold files created")


# ============================================================
# 6. TRAIN NN4B USING FOLD-SPECIFIC INTERVAL DATA
# ============================================================

X_final = np.asarray(soap.create(structures_final, n_jobs=-1))

fold_results = []

for fold in range(1, 6):

    print(f"\n==================== FOLD {fold}/5 ====================")

    vivo_train = read(f"in_vivo_80_{fold:02d}_train.xyz", index=":")
    vivo_val = read(f"in_vivo_20_{fold:02d}_test.xyz", index=":")

    interval_train = read(f"interval_80_{fold:02d}_train.xyz", index=":")
    interval_val = read(f"interval_20_{fold:02d}_test.xyz", index=":")

    train_structures = vivo_train + interval_train
    val_structures = vivo_val + interval_val

    y_train = get_energies(train_structures)
    y_val = get_energies(val_structures)

    X_train = np.asarray(soap.create(train_structures, n_jobs=-1))
    X_val = np.asarray(soap.create(val_structures, n_jobs=-1))

    model = build_model(X_train.shape[1])

    best_path = f"NN4B_fold{fold}_best.keras"
    final_path = f"NN4B_fold{fold}_final.keras"
    csv_path = f"in_vitro_20_NN4B_fold{fold}.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=best_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    model.save(final_path)

    model = tf.keras.models.load_model(best_path)

    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    final_mse = model.evaluate(X_final, y_final, verbose=0)

    y_final_pred = model.predict(X_final, verbose=0).reshape(-1)

    pd.DataFrame({
        "name": final_names,
        "expected": y_final,
        "predicted": y_final_pred
    }).to_csv(csv_path, sep=";", index=False)

    print(f"Train MSE: {train_mse:.3f}")
    print(f"Val   MSE: {val_mse:.3f}")
    print(f"Test  MSE: {final_mse:.3f}")

    fold_results.append([train_mse, val_mse, final_mse])


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

fold_results = np.array(fold_results)

np.save("NN4B_interval_5fold_results.npy", fold_results)

print("\n================ FINAL RESULTS ================")
print(f"Train: {fold_results[:, 0].mean():.3f}  {fold_results[:, 0].std():.3f}")
print(f"Val:   {fold_results[:, 1].mean():.3f}  {fold_results[:, 1].std():.3f}")
print(f"Test:  {fold_results[:, 2].mean():.3f}  {fold_results[:, 2].std():.3f}")