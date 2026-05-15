# ============================================================
# random_CV_split.py
from ase.io import read, write
from sklearn.utils import shuffle

# ============================================================
# INITIAL 80/20 DATASET SPLIT
# ============================================================
# This script was used to split the original datasets
# in_vivo.xyz and in_vitro.xyz into 80% training and 20% test parts.
#
# The resulting files were named:
#   in_vivo_80_train.xyz
#   in_vivo_20_test.xyz
#   in_vitro_80_train.xyz
#   in_vitro_20_test.xyz
#
# The in_vitro_20_test.xyz dataset was not further divided into folds.
# Only the 80% training parts were later split into five folds.

#datasets = {
#    "in_vivo": "in_vivo.xyz",
#    "in_vitro": "in_vitro.xyz"
#}

#train_ratio = 0.80
#random_state = 42

#for dataset_name, file_name in datasets.items():
#    structures = read(file_name, index=":")
#    print(f"{dataset_name}: total number of molecules = {len(structures)}")
#
#    structures = shuffle(structures, random_state=random_state)

#    n_structures = len(structures)
#    n_train = int(n_structures * train_ratio)
#
#    train_structures = structures[:n_train]
#    test_structures = structures[n_train:]
#
#    train_file = f"{dataset_name}_80_train.xyz"
#    test_file = f"{dataset_name}_20_test.xyz"

#    write(train_file, train_structures)
#    write(test_file, test_structures)

#    print(
#        f"{dataset_name}: "
#        f"train={len(train_structures)}, "
#        f"test={len(test_structures)}"
#    )


# ============================================================
# FIVE-FOLD SPLIT OF THE 80% TRAINING DATASET
# ============================================================
# After the original datasets were first divided into 80% training
# and 20% test subsets, the 80% training part was further split
# into five folds.
#
# This step was used to create five train/test combinations
# for cross-validation. In each fold, approximately 20% of the
# 80% training dataset was used as the test fold, while the
# remaining 80% was used for training.
#
# The independent 20% test dataset, e.g. in_vitro_20_test.xyz,
# was not divided into folds.

file_name = "in_vitro_80_train.xyz"

structures = read(file_name, index=":")
print(f"Number of molecules: {len(structures)}")

# Shuffle only once to ensure reproducibility
structures = shuffle(structures, random_state=42)

n_structures = len(structures)
n_folds = 5
fold_size = n_structures // n_folds

print(f"Fold size: {fold_size}")

for i in range(n_folds):
    start = i * fold_size
    end = (i + 1) * fold_size if i < n_folds - 1 else n_structures

    test_structures = structures[start:end]
    train_structures = structures[:start] + structures[end:]

    fold_id = i + 1

    train_file = f"in_vitro_80_{fold_id:02d}_train.xyz"
    test_file = f"in_vitro_80_{fold_id:02d}_test.xyz"

    write(train_file, train_structures)
    write(test_file, test_structures)

    print(
        f"Fold {fold_id}: "
        f"train={len(train_structures)}, "
        f"test={len(test_structures)}"
    )