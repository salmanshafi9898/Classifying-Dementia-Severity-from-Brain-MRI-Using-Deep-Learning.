# Complete DataLoaders
# Parameters
batch_size = 32
num_workers = 2
pin_memory = True if torch.cuda.is_available() else False

# If train_dataset is defined and has attribute .targets or you used a df, we compute class weights.
def compute_sampler_from_dataset(ds):
    """Return a WeightedRandomSampler or None if not possible."""
    # try several ways to access labels
    labels = None
    if hasattr(ds, 'targets'):
        labels = ds.targets
    else:
        # try reading labels by iterating once (safe for small datasets)
        try:
            labels = [ds[i][1] for i in range(len(ds))]
        except Exception as e:
            print("Warning: couldn't read labels to build sampler:", e)
            labels = None

    if labels is None:
        return None

    labels = list(labels)
    classes, counts = np.unique(labels, return_counts=True)
    print("Classes found:", classes, "Counts:", counts)
    # If perfectly balanced, no need for sampler
    if len(counts) <= 1:
        return None

    class_weights = 1.0 / (counts + 1e-12)
    sample_weights = np.array([class_weights[int(l)] for l in labels], dtype=np.float64)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# Build sampler if possible
sampler = None
try:
    sampler = compute_sampler_from_dataset(train_dataset)
except NameError:
    raise RuntimeError("train_dataset not found. Ensure earlier cells created train_dataset, val_dataset, test_dataset.")

# Create dataloaders; if sampler present, don't set shuffle=True for train loader
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          sampler=sampler, shuffle=(sampler is None),
                          num_workers=num_workers, pin_memory=pin_memory)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

print("Train loader batches:", len(train_loader), "Val loader batches:", len(val_loader), "Test loader batches:", len(test_loader))

batch_imgs, batch_labels = next(iter(train_loader))
print("One batch shapes -> images:", batch_imgs.shape, "labels:", batch_labels.shape)
