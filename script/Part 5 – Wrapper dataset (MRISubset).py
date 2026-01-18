""" Part 5 – Wrapper dataset (MRISubset)
base_dataset contains all slices and labels; train_indices, val_indices, test_indices tell us which slice indices belong to each split.

We define a custom dataset MRISubset that:

Stores:

the base_dataset,

a list of indices (which slices to use),

and a transform (how to preprocess each image).

len → returns how many slices are in this subset.

getitem(i):

looks up the real index self.indices[i] inside base_dataset,

gets (img, label) for that slice,

applies the appropriate transform,

returns (transformed_img, label).

We then create:

train_dataset using train_indices + train_transform (with augmentation),

val_dataset and test_dataset using their indices + val_test_transform (no augmentation).

The print at the end confirms how many slices are in each split. """

# Wrapper Dataset
class MRISubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base_dataset[self.indices[i]]  # img is PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, label

train_dataset = MRISubset(base_dataset, train_indices, train_transform)
val_dataset   = MRISubset(base_dataset, val_indices,   val_test_transform)
test_dataset  = MRISubset(base_dataset, test_indices,  val_test_transform)

print("Final sizes (slices) -> "
      f"train: {len(train_dataset)}, "
      f"val: {len(val_dataset)}, "
      f"test: {len(test_dataset)}")
