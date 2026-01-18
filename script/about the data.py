"""About the dataset
The Kaggle dataset is a preprocessed 2D image version of the original OASIS-1 brain MRI dataset. It’s built from the Open Access Series of Imaging Studies (OASIS) – a well-known public MRI dataset of young, middle-aged, and older adults, both non-demented and demented.The Kaggle version converts the original 3D MRI volumes into many 2D axial slices for deep learning.

The dementia labels come from the Clinical Dementia Rating (CDR) scale and are grouped into 4 categories:

Non-demented
Very mild dementia
Mild dementia
Moderate dementia
About 80,000 MRI slices in total from around 461 patients. Slices are taken from the middle portion of the brain volume (e.g. slice indices 100–160) to focus on informative regions They organized them into 4 class folders (one per condition) when preparing for training.

So, it’s a big set of 2D brain slices, not raw 3D medical volumes."""

# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import shutil

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

from google.colab import drive
drive.mount('/content/drive')

