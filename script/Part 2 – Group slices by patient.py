""" 
Why? Each patient has ~60 slices. If we randomly split at the slice level, slices from the same patient could end up in both train and test → data leakage and over-optimistic accuracy.

So in Part 2 we:

Loop over every (filepath, label) in base_dataset.samples. Example filename: OAS1_0028_MR1_mpr-1_100.jpg

For each file: Take just the filename: OAS1_0028_MR1_mpr-1_100.jpg.

Split by underscores: ['OAS1', '0028', 'MR1', 'mpr-1', '100.jpg']

Define the patient ID as the first 3 parts joined: patient_id = "OAS1_0028_MR1". All slices whose name starts with OAS1_0028_MR1_... belong to the same patient.

We build three data structures:

patient_to_indices[patient_id] → list of slice indices for that patient

e.g. patient_to_indices['OAS1_0028_MR1'] = [0, 1, 2, ..., 59]

patient_label[patient_id] → class label (0–3) for that patient

We also check that all slices for a patient have the same label; if not, we raise an error.

class_to_patients[label] → set of patient IDs that belong to each class
e.g. class_to_patients[0] = all Mild Dementia patient IDs.
We then print:

Total number of unique patients.

For each label: number of patients in that class.

Goal of Part 2: Stop thinking of the dataset as “independent slices” and instead as patients with groups of slices, so we can do a subject-level train/val/test split."""

# Group by patient
patient_to_indices = defaultdict(list)   # patient_id -> list of slice indices
patient_label      = dict()             # patient_id -> class label (0..3)
class_to_patients  = defaultdict(set)   # label -> set of patient_ids

for idx, (filepath, label) in enumerate(base_dataset.samples):
    fname = os.path.basename(filepath)
    parts = fname.split('_')

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {fname}")

    # Patient/session ID is the first 3 chunks: 'OAS1_0028_MR1'
    patient_id = "_".join(parts[:3])
    patient_to_indices[patient_id].append(idx)

    # Check that all slices of this patient have the same class label
    if patient_id in patient_label and patient_label[patient_id] != label:
        raise ValueError(f"Patient {patient_id} has inconsistent labels "
                         f"{patient_label[patient_id]} vs {label}")
    patient_label[patient_id] = label
    class_to_patients[label].add(patient_id)

print("Total unique patients:", len(patient_to_indices))
for lbl, pats in class_to_patients.items():
    print(f"Label {lbl} ({base_dataset.classes[lbl]}): {len(pats)} patients")
