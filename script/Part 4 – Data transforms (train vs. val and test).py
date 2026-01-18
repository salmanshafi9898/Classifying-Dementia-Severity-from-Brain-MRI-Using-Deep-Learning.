"""We define image transforms to:

Make all images the same size (224×224),

Convert them to PyTorch tensors,

Normalize pixel values for more stable training,

And (for train only) add light data augmentation.

train_transform:

Resize(224, 224) → standard input size for the model.

RandomRotation(±10°) + fill=0 → small realistic head tilt, background stays black.

RandomHorizontalFlip(p=0.5) → optional left–right flip to increase variability.

ColorJitter → slight brightness/contrast changes to mimic scanner differences.

ToTensor() → converts PIL image to tensor in [0, 1].

Normalize(mean, std) → rescales values (here using [0.5, 0.5, 0.5] as a simple default).

val_test_transform:

Only Resize, ToTensor, and Normalize (no randomness).

Ensures validation and test results are consistent and comparable."""

# Transforms
norm_mean = [0.5, 0.5, 0.5]
norm_std  = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(
        degrees=10,
        fill=0
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])
