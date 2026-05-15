# -*- coding: utf-8 -*-

from ase.io import read
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dscribe.descriptors import SOAP


# ============================================================
# NN2B_random_enrichment_5fold_final_invitro_test.py
# ============================================================
# This script trains the NN2B random enrichment model using
# five-fold cross-validation.
#
# In each fold, the training set consists of:
#   in_vivo_80_XX_train.xyz
#   in_vitro_80_XX_train.xyz
#
# The validation set consists of:
#   in_vivo_20_XX_test.xyz
#   in_vitro_80_XX_test.xyz
#
# The independent final in vitro test set:
#   in_vitro_20_test.xyz
#
# is kept untouched during training and validation. It is used only
# for the final evaluation of each trained fold model.
#
# Input and output file names are kept unchanged.
# ============================================================


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
    """Create the NN2B neural network architecture."""
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


print(">> Starting NN2B Random Enrichment Model")


# ============================================================
# 1. FINAL INDEPENDENT TEST SET
# ============================================================

print(">> Loading final independent in-vitro test set (20%)")

structures_final = read("in_vitro_20_test.xyz", index=":")
y_final = get_energies(structures_final)
final_names = get_names(structures_final)

print(f">> Final test size: {len(structures_final)}")


# ============================================================
# 2. LOAD ALL DATA FOR SOAP SPECIES DEFINITION
# ============================================================

print(">> Loading all structures to determine SOAP species")

all_structures = []

for i in range(1, 6):
    all_structures += read(f"in_vivo_80_{i:02d}_train.xyz", index=":")
    all_structures += read(f"in_vivo_20_{i:02d}_test.xyz", index=":")
    all_structures += read(f"in_vitro_80_{i:02d}_train.xyz", index=":")
    all_structures += read(f"in_vitro_80_{i:02d}_test.xyz", index=":")

species = sorted(set(
    atom
    for structure in all_structures
    for atom in structure.get_chemical_symbols()
))

print(f">> SOAP species: {species}")


# ============================================================
# 3. SOAP DESCRIPTOR
# ============================================================

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

X_final = np.asarray(soap.create(structures_final, n_jobs=-1))

print(f">> SOAP dimension: {X_final.shape[1]}")


# ============================================================
# 4. FIVE-FOLD CROSS-VALIDATION
# ============================================================

fold_results = []

for fold in range(1, 6):

    print(f"\n==================== FOLD {fold}/5 ====================")

    vivo_train = read(f"in_vivo_80_{fold:02d}_train.xyz", index=":")
    vivo_val = read(f"in_vivo_20_{fold:02d}_test.xyz", index=":")

    vitro_train = read(f"in_vitro_80_{fold:02d}_train.xyz", index=":")
    vitro_val = read(f"in_vitro_80_{fold:02d}_test.xyz", index=":")

    train_structures = vivo_train + vitro_train
    val_structures = vivo_val + vitro_val

    print(f">> Train size: {len(train_structures)}")
    print(f">> Validation size: {len(val_structures)}")

    y_train = get_energies(train_structures)
    y_val = get_energies(val_structures)

    X_train = np.asarray(soap.create(train_structures, n_jobs=-1))
    X_val = np.asarray(soap.create(val_structures, n_jobs=-1))

    model = build_model(X_train.shape[1])

    best_model_path = f"NN2B_fold{fold}_best.keras"
    final_model_path = f"NN2B_fold{fold}_final.keras"
    csv_name = f"in_vitro_20_NN2B_fold{fold}.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
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

    model.save(final_model_path)
    print(f">> Saved final model: {final_model_path}")

    model = tf.keras.models.load_model(best_model_path)

    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    final_mse = model.evaluate(X_final, y_final, verbose=0)

    y_final_pred = model.predict(X_final, verbose=0).reshape(-1)

    pd.DataFrame({
        "name": final_names,
        "expected": y_final,
        "predicted": y_final_pred
    }).to_csv(csv_name, sep=";", index=False)

    print(f">> Saved predictions: {csv_name}")

    print(f"Fold {fold} MSE:")
    print(f"  Train:      {train_mse:.3f}")
    print(f"  Validation: {val_mse:.3f}")
    print(f"  Final test: {final_mse:.3f}")

    fold_results.append([train_mse, val_mse, final_mse])


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

fold_results = np.array(fold_results)

print("\n==================== FINAL RESULTS ====================")
print(f"Train MSE:      {fold_results[:, 0].mean():.3f}  {fold_results[:, 0].std():.3f}")
print(f"Val MSE:        {fold_results[:, 1].mean():.3f}  {fold_results[:, 1].std():.3f}")
print(f"Final Test MSE: {fold_results[:, 2].mean():.3f}  {fold_results[:, 2].std():.3f}")

np.save("NN2B_random_enrichment_5fold_results.npy", fold_results)