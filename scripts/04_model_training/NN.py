# -*- coding: utf-8 -*-

from ase.io import read, write
import tensorflow as tf
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold


# ============================================================
# NN_invitro_5fold_crossvalidation.py
# ============================================================
# This script performs the final five-fold cross-validation
# of the NN model using the in vitro dataset.
#
# The dataset:
#   in_vitro_80_train.xyz
#
# is divided into five cross-validation folds:
#   nn_train_foldXX.xyz
#   nn_val_foldXX.xyz
#
# In each fold, the model is trained on four parts of the
# in vitro pool and validated on the remaining part.
#
# The independent final in vitro test set:
#   in_vitro_20_test.xyz
#
# is kept untouched during cross-validation. It is used only
# for the final evaluation of each trained fold model.
#
# Input and output file names are kept unchanged.
# ============================================================


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
    """Create the NN neural network architecture."""
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


print(">> Starting NN: five-fold cross-validation")


# ============================================================
# 1. LOAD DATA
# ============================================================

structures_final = read("in_vitro_20_test.xyz", index=":")
vitro_pool = read("in_vitro_80_train.xyz", index=":")

y_final = get_energies(structures_final)
final_names = get_names(structures_final)

print(f">> Final test size: {len(structures_final)}")
print(f">> Pool size for CV: {len(vitro_pool)}")


# ============================================================
# 2. SOAP SPECIES
# ============================================================

print(">> Determining SOAP species")

all_structures = vitro_pool + structures_final

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

X_final = np.asarray(soap.create(structures_final, n_jobs=-1))


# ============================================================
# 3. CREATE FIVE CROSS-VALIDATION FILES
# ============================================================

print(">> Creating five-fold CV split files")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

for fold_id, (train_idx, test_idx) in enumerate(kf.split(vitro_pool), start=1):

    fold_train = [vitro_pool[i] for i in train_idx]
    fold_val = [vitro_pool[i] for i in test_idx]

    write(f"nn_train_fold{fold_id:02d}.xyz", fold_train)
    write(f"nn_val_fold{fold_id:02d}.xyz", fold_val)


# ============================================================
# 4. TRAINING LOOP
# ============================================================

fold_results = []

for fold in range(1, 6):

    print(f"\n==================== FOLD {fold}/5 ====================")

    train_structures = read(f"nn_train_fold{fold:02d}.xyz", index=":")
    val_structures = read(f"nn_val_fold{fold:02d}.xyz", index=":")

    y_train = get_energies(train_structures)
    y_val = get_energies(val_structures)

    X_train = np.asarray(soap.create(train_structures, n_jobs=-1))
    X_val = np.asarray(soap.create(val_structures, n_jobs=-1))

    model = build_model(X_train.shape[1])

    best_path = f"NN_fold{fold}_best.keras"

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

    model = tf.keras.models.load_model(best_path)

    train_mse = model.evaluate(X_train, y_train, verbose=0)
    val_mse = model.evaluate(X_val, y_val, verbose=0)
    final_mse = model.evaluate(X_final, y_final, verbose=0)

    y_final_pred = model.predict(X_final, verbose=0).reshape(-1)

    pd.DataFrame({
        "name": final_names,
        "expected": y_final,
        "predicted": y_final_pred
    }).to_csv(f"in_vitro_20_NN_fold{fold}.csv", sep=";", index=False)

    print(
        f"Fold {fold} Results -> "
        f"Train: {train_mse:.3f}, "
        f"Val: {val_mse:.3f}, "
        f"Test: {final_mse:.3f}"
    )

    fold_results.append([train_mse, val_mse, final_mse])


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

fold_results = np.array(fold_results)

np.save("NN_cv_results.npy", fold_results)

print("\n================ FINAL CROSS-VALIDATION SUMMARY ================")
print(f"Train MSE: {fold_results[:, 0].mean():.4f}  {fold_results[:, 0].std():.4f}")
print(f"Val MSE:   {fold_results[:, 1].mean():.4f}  {fold_results[:, 1].std():.4f}")
print(f"Test MSE:  {fold_results[:, 2].mean():.4f}  {fold_results[:, 2].std():.4f}")