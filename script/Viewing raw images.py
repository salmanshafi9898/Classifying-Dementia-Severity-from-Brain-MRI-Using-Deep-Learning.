""" Viewing raw images
Let's view what our images look like raw so we can decide on how to transform and normalize them."""

sample_dir = "data/Data/Non Demented"   # pick any class folder
sample_files = os.listdir(sample_dir)
print("Num files in sample_dir:", len(sample_files))
print("First few files:", sample_files[:5])

# Open one image
img_path = os.path.join(sample_dir, sample_files[0])
img = Image.open(img_path)
arr = np.array(img)

print("PIL mode:", img.mode)          # 'L' = grayscale, 'RGB' = color
print("Array shape:", arr.shape)      # (H, W) or (H, W, C)
print("Pixel min/max:", arr.min(), arr.max())

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


