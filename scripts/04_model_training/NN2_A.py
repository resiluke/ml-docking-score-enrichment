# -*- coding: utf-8 -*-

from ase.io import read
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dscribe.descriptors import SOAP


# ============================================================
# NN2A_in_vitro_5fold_final_invitro_test.py
# ============================================================
# This script trains the NN2A neural network model using five-fold
# cross-validation on the in vitro dataset.
#
# In each fold, the model is trained on:
#   in_vitro_80_XX_train.xyz
#
# and validated on:
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
    """Create the NN2A neural network architecture."""
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


# ============================================================
# 1. FINAL TEST SET (independent, untouched)
# ============================================================

print(">> Loading FINAL in-vitro test data (20%)...")

structures_final = read("in_vitro_20_test.xyz", index=":")
y_final = get_energies(structures_final)
final_names = get_names(structures_final)

print(f">> Final test size: {len(structures_final)}")


# ============================================================
# 2. LOAD ALL IN-VITRO DATA FOR SOAP SPECIES DEFINITION
# ============================================================

print(">> Loading all in-vitro data for SOAP species...")

all_structures = []

for i in range(1, 6):
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

print(">> Initializing SOAP...")

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
# 4. FIVE-FOLD CROSS-VALIDATION ON IN VITRO DATA
# ============================================================

fold_results = []

for fold in range(1, 6):

    print(f"\n==================== FOLD {fold}/5 ====================")

    train_file = f"in_vitro_80_{fold:02d}_train.xyz"
    val_file = f"in_vitro_80_{fold:02d}_test.xyz"

    checkpoint_path = f"NN2A_fold{fold}_best.keras"
    final_model_path = f"NN2A_fold{fold}_final.keras"
    csv_name = f"in_vitro_20_NN2A_fold{fold}.csv"

    structures_train = read(train_file, index=":")
    structures_val = read(val_file, index=":")

    y_train = get_energies(structures_train)
    y_val = get_energies(structures_val)

    X_train = np.asarray(soap.create(structures_train, n_jobs=-1))
    X_val = np.asarray(soap.create(structures_val, n_jobs=-1))

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

    model = build_model(X_train.shape[1])

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
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

    model = tf.keras.models.load_model(checkpoint_path)

    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    final_mse = model.evaluate(X_final, y_final, verbose=0)

    y_final_pred = model.predict(X_final, verbose=0).reshape(-1)

    pd.DataFrame({
        "name": final_names,
        "expected": y_final,
        "predicted": y_final_pred
    }).to_csv(csv_name, sep=";", index=False)

    print(f">> Saved final test predictions: {csv_name}")

    print(f"Fold {fold} MSE:")
    print(f"  Train:       {train_mse:.3f}")
    print(f"  Validation:  {val_mse:.3f}")
    print(f"  Final test:  {final_mse:.3f}")

    fold_results.append([train_mse, val_mse, final_mse])


# ============================================================
# 5. FINAL RESULTS
# ============================================================

fold_results = np.array(fold_results)

print("\n==================== FINAL RESULTS ====================")
print(f"Train MSE:      {fold_results[:, 0].mean():.3f}  {fold_results[:, 0].std():.3f}")
print(f"Val MSE:        {fold_results[:, 1].mean():.3f}  {fold_results[:, 1].std():.3f}")
print(f"Final Test MSE: {fold_results[:, 2].mean():.3f}  {fold_results[:, 2].std():.3f}")

np.save("NN2A_in_vitro_5fold_results.npy", fold_results)