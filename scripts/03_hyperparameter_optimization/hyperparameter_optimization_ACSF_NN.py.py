# -*- coding: utf-8 -*-

import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from ase.io import read
from dscribe.descriptors import ACSF
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# hyperparameter_optimization_ACSF_NN.py
# ============================================================
# This script performs hyperparameter optimization of a neural
# network model using ACSF molecular descriptors.
#
# The optimization is performed on a subset of the in_vivo.xyz
# dataset. For each trial, ACSF descriptors are calculated,
# averaged over atoms, scaled, and then used as input to a neural
# network model.
#
# The resulting MSE on the test set is used as the objective
# function.
#
# The best model is saved as:
#   best_loss_model_ACSF.keras
#
# The optimization results are exported to:
#   hyperopt_results_ACSF_NN.csv
# ============================================================


FILE_NAME = "in_vivo.xyz"
N_STRUCTURES = 10000

RANDOM_SEED = 42
MAX_EVALS = 50

BEST_MODEL_FILE = "best_loss_model_ACSF.keras"
CHECKPOINT_FILE = "best_model_ACSF.keras"

X_TEST_FILE = "x_test_ACSF.npy"
Y_TEST_FILE = "y_test_ACSF.npy"

RESULTS_CSV = "hyperopt_results_ACSF_NN.csv"
PLOT_FILE = "hyperopt_best_model_ACSF_prediction.png"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


print(f">> Loading data from: {FILE_NAME}")

structures = read(FILE_NAME, index=f":{N_STRUCTURES}")

try:
    targets = np.array([
        float(structure.get_potential_energy())
        for structure in structures
    ])
except Exception as error:
    raise RuntimeError(f"Could not load energies: {error}")

species = sorted(set(
    atom
    for structure in structures
    for atom in structure.get_chemical_symbols()
))

print(f">> Number of molecules: {len(structures)}")
print(f">> ACSF species: {species}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

structures_train, structures_tmp, y_train, y_tmp = train_test_split(
    structures,
    targets,
    test_size=0.25,
    random_state=RANDOM_SEED
)

structures_val, structures_test, y_val, y_test = train_test_split(
    structures_tmp,
    y_tmp,
    test_size=0.40,
    random_state=RANDOM_SEED
)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

print(
    f">> Split sizes: "
    f"Train={len(structures_train)}, "
    f"Validation={len(structures_val)}, "
    f"Test={len(structures_test)}"
)


# ============================================================
# MODEL DEFINITION
# ============================================================

def build_model(input_dim, activation):
    """Build neural network model."""

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(100, activation=activation),
        tf.keras.layers.Dense(50, activation=activation),
        tf.keras.layers.Dense(5, activation=activation),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    return model


def get_optimizer(name):
    """Return optimizer by name."""

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=1e-3)

    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=1e-3,
            momentum=0.9
        )

    raise ValueError(f"Unknown optimizer: {name}")


def create_acsf_descriptors(acsf, structures_subset):
    """Create molecule-level ACSF descriptors by averaging atomic ACSF vectors."""

    descriptors = []

    for structure in structures_subset:
        atomic_descriptors = acsf.create(
            structure,
            n_jobs=os.cpu_count()
        )

        descriptors.append(
            atomic_descriptors.mean(axis=0)
        )

    return np.asarray(descriptors)


# ============================================================
# HYPEROPT OBJECTIVE FUNCTION
# ============================================================

best_loss = [np.inf]


def objective(params):
    """Objective function minimized by Hyperopt."""

    try:
        print(f"\n>> Testing parameters: {params}")

        acsf = ACSF(
            species=species,
            r_cut=params["r_cut"],
            g2_params=[
                [params["g2_eta"], params["g2_rs"]]
            ],
            g4_params=[
                [
                    params["g4_eta"],
                    params["g4_zeta"],
                    params["g4_lambda"]
                ]
            ],
            periodic=False
        )

        start_descriptor = time.time()

        x_train = create_acsf_descriptors(acsf, structures_train)
        x_val = create_acsf_descriptors(acsf, structures_val)
        x_test = create_acsf_descriptors(acsf, structures_test)

        descriptor_time = time.time() - start_descriptor

        print(f">> ACSF descriptor calculation time: {descriptor_time:.2f} s")

        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        model = build_model(
            input_dim=x_train.shape[1],
            activation=params["activation"]
        )

        model.compile(
            optimizer=get_optimizer(params["optimizer"]),
            loss="mean_squared_error",
            metrics=["mean_squared_error"]
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILE,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=params["batch_size"],
            callbacks=[checkpoint, early_stopping],
            verbose=0
        )

        model = tf.keras.models.load_model(CHECKPOINT_FILE)

        mse = model.evaluate(x_test, y_test, verbose=0)[0]

        descriptor_size = x_train.shape[1]
        parameter_count = model.count_params()

        print(f">> Test MSE: {mse:.4f}")

        if mse < best_loss[0]:
            print(f">> New best model saved with MSE = {mse:.4f}")

            best_loss[0] = mse

            model.save(BEST_MODEL_FILE)
            np.save(X_TEST_FILE, x_test)
            np.save(Y_TEST_FILE, y_test)

        tf.keras.backend.clear_session()
        gc.collect()

        return {
            "loss": mse,
            "status": STATUS_OK,
            "descriptor_time": descriptor_time,
            "descriptor_size": descriptor_size,
            "parameter_count": parameter_count
        }

    except Exception as error:
        print(f"[ERROR] {error}")

        tf.keras.backend.clear_session()
        gc.collect()

        return {
            "loss": 1e6,
            "status": STATUS_OK
        }


# ============================================================
# HYPERPARAMETER SPACE
# ============================================================

space = {
    "r_cut": hp.uniform("r_cut", 7, 11),
    "g2_eta": hp.uniform("g2_eta", 0.09, 0.60),
    "g2_rs": hp.uniform("g2_rs", 2, 5),
    "g4_eta": hp.uniform("g4_eta", 0.10, 2.00),
    "g4_zeta": hp.quniform("g4_zeta", 7, 10, 1),
    "g4_lambda": hp.choice("g4_lambda", [-1, 1]),
    "batch_size": hp.choice("batch_size", [32, 64]),
    "activation": hp.choice("activation", ["relu", "softplus", "sigmoid", "tanh"]),
    "optimizer": hp.choice("optimizer", ["adam", "sgd"]),
}


# ============================================================
# RUN OPTIMIZATION
# ============================================================

start_time = time.time()

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=trials,
    rstate=np.random.default_rng(RANDOM_SEED)
)

elapsed_time = time.time() - start_time

print("\n>> Best Hyperopt indices/values:")
print(best)


# ============================================================
# EXPORT RESULTS
# ============================================================

def export_trials_to_csv(trials, elapsed_time, filename):
    """Export Hyperopt trials to CSV."""

    records = []

    batch_size_values = [32, 64]
    activation_values = ["relu", "softplus", "sigmoid", "tanh"]
    optimizer_values = ["adam", "sgd"]
    lambda_values = [-1, 1]

    for trial in trials.trials:

        vals = trial["misc"]["vals"]
        result = trial["result"]

        row = {
            "r_cut": vals["r_cut"][0],
            "g2_eta": vals["g2_eta"][0],
            "g2_rs": vals["g2_rs"][0],
            "g4_eta": vals["g4_eta"][0],
            "g4_zeta": int(vals["g4_zeta"][0]),
            "g4_lambda": lambda_values[vals["g4_lambda"][0]],
            "batch_size": batch_size_values[vals["batch_size"][0]],
            "activation": activation_values[vals["activation"][0]],
            "optimizer": optimizer_values[vals["optimizer"][0]],
            "mse": result.get("loss", np.nan),
            "total_time_sec": elapsed_time,
            "total_time_min": elapsed_time / 60,
            "descriptor_time": result.get("descriptor_time", np.nan),
            "descriptor_size": result.get("descriptor_size", np.nan),
            "parameter_count": result.get("parameter_count", np.nan),
        }

        records.append(row)

    df_trials = pd.DataFrame(records)
    df_trials.to_csv(filename, sep=";", index=False)

    print(f">> Optimization results saved to: {filename}")


export_trials_to_csv(
    trials,
    elapsed_time,
    RESULTS_CSV
)


# ============================================================
# BEST MODEL PREDICTION PLOT
# ============================================================

if (
    os.path.exists(BEST_MODEL_FILE) and
    os.path.exists(X_TEST_FILE) and
    os.path.exists(Y_TEST_FILE)
):

    print("\n>> Creating prediction plot for the best model")

    model = tf.keras.models.load_model(BEST_MODEL_FILE)

    x_test = np.load(X_TEST_FILE)
    y_test_saved = np.load(Y_TEST_FILE)

    y_pred = model.predict(x_test, verbose=0).reshape(-1)
    y_true = y_test_saved.reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    regression = LinearRegression().fit(
        y_true.reshape(-1, 1),
        y_pred
    )

    slope = regression.coef_[0]
    intercept = regression.intercept_

    regression_line = regression.predict(
        y_true.reshape(-1, 1)
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        c="magenta",
        s=10,
        alpha=0.6,
        label="Predictions"
    )

    plt.plot(
        y_true,
        y_true,
        color="black",
        linewidth=1,
        label="Ideal line: y = x"
    )

    plt.plot(
        y_true,
        regression_line,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Regression: y = {slope:.2f}x + {intercept:.2f}"
    )

    plt.xlabel("Expected values")
    plt.ylabel("Predicted values")
    plt.title("Prediction vs expected values")

    plt.legend(loc="upper left")

    plt.text(
        0.97,
        0.03,
        f"MSE = {mse:.3f}\nR² = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        ),
        ha="right",
        va="bottom"
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=600, bbox_inches="tight")
    plt.close()

    print(f">> Prediction plot saved to: {PLOT_FILE}")

else:
    print("\n[SKIP] Best model or saved test data not found.")