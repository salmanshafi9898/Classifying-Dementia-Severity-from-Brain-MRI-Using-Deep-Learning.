/* We use ImageFolder on data/MRI_IMAGES, which has 4 folders: Mild Dementia, Moderate Dementia, Non Demented, Very mild Dementia.

ImageFolder:

Treats each folder name as a class label.

Assigns them integer IDs, e.g. Mild Dementia → 0, Moderate Dementia → 1, Non Demented → 2, Very mild Dementia → 3

It builds one big dataset called base_dataset:

Conceptually: [ (slice_0_path, label_0), (slice_1_path, label_1), ..., (slice_N_path, label_N) ]

Each element = one MRI slice and its class label.

We print:

the class names (base_dataset.classes)

the mapping folder → label ID (class_to_idx)

the total number of slices (len(base_dataset))

one example sample to confirm the filename format.

Goal of Part 1: Create a single object that contains all slices + labels, so we can then organize them by patient.*/

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

data_dir = "data/Data"   # root with 4 class folders

# No transform yet – we want raw PIL images & file paths
base_dataset = datasets.ImageFolder(root=data_dir, transform=None)

print("Classes:", base_dataset.classes)
print("class_to_idx:", base_dataset.class_to_idx)
print("Total slices:", len(base_dataset))

print("Example sample:", base_dataset.samples[0])

