import os, time, copy, math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import torchvision.models as models
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def infer_num_classes(train_dataset=None, label_map=None):
    if label_map is not None:
        return len(label_map)
    if train_dataset is not None and hasattr(train_dataset, "targets"):
        return len(set(list(train_dataset.targets)))
    # fallback: infer by scanning a few batches
    y = []
    for _, labels in train_loader:
        y.extend(labels.numpy().tolist())
        if len(y) > 2000:
            break
    return len(set(y))

num_classes = infer_num_classes(
    train_dataset=globals().get("train_dataset", None),
    label_map=globals().get("label_map", None)
)
print("num_classes =", num_classes)

def compute_class_weights_from_loader(loader, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in loader:
        labels_np = labels.numpy()
        for c in range(num_classes):
            counts[c] += np.sum(labels_np == c)
    counts = np.maximum(counts, 1)
    weights = (counts.sum() / counts).astype(np.float32)  # inverse frequency
    weights = weights / weights.mean()                    # normalize
    return torch.tensor(weights, dtype=torch.float32)

class_weights = compute_class_weights_from_loader(train_loader, num_classes).to(device)
print("Class weights:", class_weights.detach().cpu().numpy())
