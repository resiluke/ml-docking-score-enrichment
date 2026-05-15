# -*- coding: utf-8 -*-

from ase.io import read, write
import tensorflow as tf
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold


# ============================================================
# NN3A_topN_selection_invitro_5fold.py
# ============================================================
# This script trains the NN3A model using TOP_N selection from
# the in vitro training dataset.
#
# First, the pre-trained NN1 fold models are used as an ensemble
# to predict docking scores for:
#   in_vitro_80_train.xyz
#
# The average ensemble prediction is used to select TOP_N compounds
# with the lowest predicted docking scores.
#
# The selected subset is saved as:
#   top_N_pool.xyz
#
# This subset is then divided into five cross-validation folds:
#   topA_80_XX_train.xyz
#   topA_20_XX_test.xyz
#
# The independent final in vitro test set:
#   in_vitro_20_test.xyz
#
# is kept untouched during selection, training, and validation. It is
# used only for the final evaluation of each trained fold model.
#
# Input and output file names are kept unchanged.
# ============================================================


TOP_N = 29942
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
    """Create the NN3A neural network architecture."""
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


print(">> NN3A in vitro TOP_N selection model")


# ============================================================
# 1. LOAD DATA
# ============================================================

print(">> Loading data")

vitro_pool = read("in_vitro_80_train.xyz", index=":")
structures_final = read("in_vitro_20_test.xyz", index=":")

y_final = get_energies(structures_final)
final_names = get_names(structures_final)


# ============================================================
# 2. SOAP SPECIES AND DESCRIPTOR
# ============================================================

print(">> Preparing SOAP species")

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


# ============================================================
# 3. NN1 ENSEMBLE PREDICTION ON IN VITRO POOL
# ============================================================

print(">> Running NN1 ensemble prediction")

X_pool = np.asarray(soap.create(vitro_pool, n_jobs=-1))

ensemble_preds = []

for fold in range(1, 6):
    model_path = f"NN1_fold{fold}_best.keras"
    print(f">> Loading {model_path}")

    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_pool, verbose=0).reshape(-1)

    ensemble_preds.append(preds)

mean_preds = np.mean(np.array(ensemble_preds), axis=0)


# ============================================================
# 4. SELECT TOP_N COMPOUNDS
# ============================================================

print(">> Selecting TOP_N compounds")

top_indices = np.argsort(mean_preds)[:TOP_N]
top_structures = [vitro_pool[i] for i in top_indices]

print(f">> Selected TOP_N = {len(top_structures)}")

write("top_N_pool.xyz", top_structures)


# ============================================================
# 5. CREATE FIVE CROSS-VALIDATION FOLDS
# ============================================================

print(">> Creating cross-validation splits")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

for fold_id, (train_idx, test_idx) in enumerate(kf.split(top_structures), start=1):

    top_train = [top_structures[i] for i in train_idx]
    top_test = [top_structures[i] for i in test_idx]

    write(f"topA_80_{fold_id:02d}_train.xyz", top_train)
    write(f"topA_20_{fold_id:02d}_test.xyz", top_test)

print(">> TOP_N cross-validation files created")


# ============================================================
# 6. TRAIN NN3A ON SELECTED TOP_N DATA
# ============================================================

print(">> Preparing final test SOAP")

X_final = np.asarray(soap.create(structures_final, n_jobs=-1))

fold_results = []

for fold in range(1, 6):

    print(f"\n==================== FOLD {fold}/5 ====================")

    train_structures = read(f"topA_80_{fold:02d}_train.xyz", index=":")
    val_structures = read(f"topA_20_{fold:02d}_test.xyz", index=":")

    y_train = get_energies(train_structures)
    y_val = get_energies(val_structures)

    X_train = np.asarray(soap.create(train_structures, n_jobs=-1))
    X_val = np.asarray(soap.create(val_structures, n_jobs=-1))

    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")

    model = build_model(X_train.shape[1])

    best_path = f"NN3A_fold{fold}_best.keras"
    final_path = f"NN3A_fold{fold}_final.keras"
    csv_path = f"in_vitro_20_NN3A_fold{fold}.csv"

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

np.save("NN3A_topDS_5fold_results.npy", fold_results)

print("\n================ FINAL RESULTS ================")
print(f"Train: {fold_results[:, 0].mean():.3f}  {fold_results[:, 0].std():.3f}")
print(f"Val:   {fold_results[:, 1].mean():.3f}  {fold_results[:, 1].std():.3f}")
print(f"Test:  {fold_results[:, 2].mean():.3f}  {fold_results[:, 2].std():.3f}")