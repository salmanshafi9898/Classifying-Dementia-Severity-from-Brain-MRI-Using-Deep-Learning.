"""
Part 3 – Patient-level 64/16/20 stratified split
We split by patient, not by slice, to avoid data leakage (no patient appears in more than one split).

For each class:

Take the list of patient IDs in that class (class_to_patients[label]), shuffle it,

Assign ~20% of patients to test, ~16% to validation, and ~64% to train.

We collect these into global lists: train_patients, val_patients, test_patients.

Then we convert patient IDs → slice indices using patient_to_indices, giving: train_indices, val_indices, test_indices.

These index lists are what we’ll use to build the final train/val/test datasets and dataloaders. """

# Stratifying by patients
random.seed(42)

train_patients = []
val_patients   = []
test_patients  = []

for lbl, patients in class_to_patients.items():
    patients = list(patients)
    random.shuffle(patients)

    n = len(patients)
    n_test = int(0.20 * n)       # 20% test
    n_dev  = n - n_test          # remaining 80% -> train+val
    n_val  = int(0.20 * n_dev)   # 20% of dev = 16% overall
    # rest (~64%) = train

    cls_test  = patients[:n_test]
    cls_val   = patients[n_test:n_test + n_val]
    cls_train = patients[n_test + n_val:]

    train_patients.extend(cls_train)
    val_patients.extend(cls_val)
    test_patients.extend(cls_test)

    print(
        f"Class {lbl} ({base_dataset.classes[lbl]}): "
        f"{len(cls_train)} train patients, "
        f"{len(cls_val)} val patients, "
        f"{len(cls_test)} test patients"
    )

print("Total train patients:", len(train_patients))
print("Total val patients:",   len(val_patients))
print("Total test patients:",  len(test_patients))

# Turn a list of patient IDs into slice indices
def patients_to_indices(patient_list):
    idxs = []
    for pid in patient_list:
        idxs.extend(patient_to_indices[pid])
    return idxs

train_indices = patients_to_indices(train_patients)
val_indices   = patients_to_indices(val_patients)
test_indices  = patients_to_indices(test_patients)

print("Total train slices:", len(train_indices))
print("Total val slices:",   len(val_indices))
print("Total test slices:",  len(test_indices))
