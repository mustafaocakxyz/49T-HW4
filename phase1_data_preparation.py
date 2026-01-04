import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# Load data
try:
    from medmnist import PneumoniaMNIST
    data_class = PneumoniaMNIST(split='train', download=True, transform=None)
    train_images = data_class.imgs
    train_labels = data_class.labels.flatten()
    data_class_val = PneumoniaMNIST(split='val', download=True, transform=None)
    val_images = data_class_val.imgs
    val_labels = data_class_val.labels.flatten()
    data_class_test = PneumoniaMNIST(split='test', download=True, transform=None)
    test_images = data_class_test.imgs
    test_labels = data_class_test.labels.flatten()
except Exception as e:
    if os.path.exists('pneumoniamnist.npz'):
        data = np.load('pneumoniamnist.npz')
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
    else:
        raise FileNotFoundError("Could not find pneumoniamnist.npz file. Please download it or install medmnist package.")

# Normalize images
train_images_norm = train_images.astype(np.float32) / 255.0
val_images_norm = val_images.astype(np.float32) / 255.0
test_images_norm = test_images.astype(np.float32) / 255.0

# Save normalized data
save_dir = "chatgpt data normalized"
os.makedirs(save_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(save_dir, 'normalized_data.npz'),
    train_images=train_images_norm,
    train_labels=train_labels,
    val_images=val_images_norm,
    val_labels=val_labels,
    test_images=test_images_norm,
    test_labels=test_labels
)

# Visualize sample images from each class
plots_dir = "phase1_plots"
os.makedirs(plots_dir, exist_ok=True)

normal_indices = np.where(train_labels == 0)[0]
pneumonia_indices = np.where(train_labels == 1)[0]
np.random.seed(42)
normal_samples = np.random.choice(normal_indices, 5, replace=False)
pneumonia_samples = np.random.choice(pneumonia_indices, 5, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Images: Normal (Top) vs Pneumonia (Bottom)', fontsize=14)

for i, idx in enumerate(normal_samples):
    axes[0, i].imshow(train_images[idx], cmap='gray')
    axes[0, i].set_title(f'Normal\nIndex: {idx}')
    axes[0, i].axis('off')

for i, idx in enumerate(pneumonia_samples):
    axes[1, i].imshow(train_images[idx], cmap='gray')
    axes[1, i].set_title(f'Pneumonia\nIndex: {idx}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot pixel intensity histograms for each class
normal_pixels = train_images[train_labels == 0].flatten()
pneumonia_pixels = train_images[train_labels == 1].flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Pixel Intensity Histograms by Class', fontsize=14)

axes[0].hist(normal_pixels, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Normal Class')
axes[0].set_xlabel('Pixel Intensity')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

axes[1].hist(pneumonia_pixels, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1].set_title('Pneumonia Class')
axes[1].set_xlabel('Pixel Intensity')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pixel_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()

# Compute and report class distributions
def compute_distribution(labels):
    total = len(labels)
    class_0_count = np.sum(labels == 0)
    class_1_count = np.sum(labels == 1)
    class_0_pct = (class_0_count / total) * 100
    class_1_pct = (class_1_count / total) * 100
    return {
        'total': total,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'class_0_pct': class_0_pct,
        'class_1_pct': class_1_pct
    }

train_dist = compute_distribution(train_labels)
val_dist = compute_distribution(val_labels)
test_dist = compute_distribution(test_labels)

print("Class Distribution:")
print(f"\nTraining Set:")
print(f"  Total samples: {train_dist['total']}")
print(f"  Normal (0): {train_dist['class_0_count']} ({train_dist['class_0_pct']:.2f}%)")
print(f"  Pneumonia (1): {train_dist['class_1_count']} ({train_dist['class_1_pct']:.2f}%)")

print(f"\nValidation Set:")
print(f"  Total samples: {val_dist['total']}")
print(f"  Normal (0): {val_dist['class_0_count']} ({val_dist['class_0_pct']:.2f}%)")
print(f"  Pneumonia (1): {val_dist['class_1_count']} ({val_dist['class_1_pct']:.2f}%)")

print(f"\nTest Set:")
print(f"  Total samples: {test_dist['total']}")
print(f"  Normal (0): {test_dist['class_0_count']} ({test_dist['class_0_pct']:.2f}%)")
print(f"  Pneumonia (1): {test_dist['class_1_count']} ({test_dist['class_1_pct']:.2f}%)")

# Calculate and save class weights
weight_class_0 = train_dist['total'] / (2 * train_dist['class_0_count'])
weight_class_1 = train_dist['total'] / (2 * train_dist['class_1_count'])

np.savez(
    os.path.join(save_dir, 'class_weights.npz'),
    weight_class_0=weight_class_0,
    weight_class_1=weight_class_1
)

