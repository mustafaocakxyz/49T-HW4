"""
Phase 1: Data Preparation and Exploration
This script handles data loading, normalization, visualization, and EDA.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from medmnist import INFO
import medmnist

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PHASE 1: DATA PREPARATION AND EXPLORATION")
print("=" * 60)

# ============================================================================
# Step 1.2: Data Loading & Initial Exploration
# ============================================================================

print("\n[Step 1.2] Loading PneumoniaMNIST dataset...")

# Load PneumoniaMNist dataset
try:
    # Try loading from medmnist package
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
    
    print("✓ Dataset loaded successfully from medmnist package")
    
except Exception as e:
    print(f"Error loading from medmnist: {e}")
    print("Attempting to load from .npz file...")
    
    # Try loading from .npz file if available
    if os.path.exists('pneumoniamnist.npz'):
        data = np.load('pneumoniamnist.npz')
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        print("✓ Dataset loaded successfully from .npz file")
    else:
        raise FileNotFoundError("Could not find pneumoniamnist.npz file. Please download it or install medmnist package.")

# Print dataset information
print("\n" + "-" * 60)
print("Dataset Shapes and Statistics:")
print("-" * 60)
print(f"Train Images Shape: {train_images.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Validation Images Shape: {val_images.shape}")
print(f"Validation Labels Shape: {val_labels.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

# Verify image dimensions
assert train_images.shape[1:] == (28, 28), f"Expected 28x28 images, got {train_images.shape[1:]}"
print(f"\n✓ Image dimensions verified: {train_images.shape[1:]} (grayscale)")

# Check label encoding
unique_labels = np.unique(train_labels)
print(f"Unique labels: {unique_labels}")
assert set(unique_labels).issubset({0, 1}), "Labels should be 0 (Normal) or 1 (Pneumonia)"
print("✓ Label encoding verified: 0=Normal, 1=Pneumonia")

# Basic statistics
print("\n" + "-" * 60)
print("Basic Statistics:")
print("-" * 60)
print(f"Train images - Min: {train_images.min()}, Max: {train_images.max()}, Mean: {train_images.mean():.2f}")
print(f"Validation images - Min: {val_images.min()}, Max: {val_images.max()}, Mean: {val_images.mean():.2f}")
print(f"Test images - Min: {test_images.min()}, Max: {test_images.max()}, Mean: {test_images.mean():.2f}")

print("\n✓ Step 1.2 completed: Data loaded and initial exploration done")

# ============================================================================
# Step 1.3: Data Normalization
# ============================================================================

print("\n[Step 1.3] Normalizing images to range [0, 1]...")

# Normalize images by dividing by 255.0
train_images_norm = train_images.astype(np.float32) / 255.0
val_images_norm = val_images.astype(np.float32) / 255.0
test_images_norm = test_images.astype(np.float32) / 255.0

# Verify normalization
print("\n" + "-" * 60)
print("Normalization Verification:")
print("-" * 60)
print(f"Train images (normalized) - Min: {train_images_norm.min():.4f}, Max: {train_images_norm.max():.4f}, Mean: {train_images_norm.mean():.4f}")
print(f"Validation images (normalized) - Min: {val_images_norm.min():.4f}, Max: {val_images_norm.max():.4f}, Mean: {val_images_norm.mean():.4f}")
print(f"Test images (normalized) - Min: {test_images_norm.min():.4f}, Max: {test_images_norm.max():.4f}, Mean: {test_images_norm.mean():.4f}")

assert train_images_norm.min() >= 0.0 and train_images_norm.max() <= 1.0, "Normalization failed: values not in [0, 1]"
print("\n✓ Normalization verified: All values in range [0, 1]")

# Create directory for normalized data
save_dir = "chatgpt data normalized"
os.makedirs(save_dir, exist_ok=True)
print(f"\n✓ Directory '{save_dir}' created/verified")

# Save normalized arrays
print(f"\nSaving normalized arrays to '{save_dir}/'...")
np.savez_compressed(
    os.path.join(save_dir, 'normalized_data.npz'),
    train_images=train_images_norm,
    train_labels=train_labels,
    val_images=val_images_norm,
    val_labels=val_labels,
    test_images=test_images_norm,
    test_labels=test_labels
)

print(f"✓ Normalized data saved to '{save_dir}/normalized_data.npz'")
print("\n✓ Step 1.3 completed: Data normalized and saved")

# ============================================================================
# Step 1.4: Data Visualization & EDA
# ============================================================================

print("\n[Step 1.4] Performing data visualization and EDA...")

# Create output directory for plots
plots_dir = "phase1_plots"
os.makedirs(plots_dir, exist_ok=True)

# 1. Visualize sample images from each class
print("\n1. Visualizing sample images from each class...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Images: Normal (Top) vs Pneumonia (Bottom)', fontsize=14)

# Get indices for each class
normal_indices = np.where(train_labels == 0)[0]
pneumonia_indices = np.where(train_labels == 1)[0]

# Sample 5 images from each class
np.random.seed(42)
normal_samples = np.random.choice(normal_indices, 5, replace=False)
pneumonia_samples = np.random.choice(pneumonia_indices, 5, replace=False)

# Plot Normal images (top row)
for i, idx in enumerate(normal_samples):
    axes[0, i].imshow(train_images[idx], cmap='gray')
    axes[0, i].set_title(f'Normal\nIndex: {idx}')
    axes[0, i].axis('off')

# Plot Pneumonia images (bottom row)
for i, idx in enumerate(pneumonia_samples):
    axes[1, i].imshow(train_images[idx], cmap='gray')
    axes[1, i].set_title(f'Pneumonia\nIndex: {idx}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
print(f"✓ Sample images saved to '{plots_dir}/sample_images.png'")
plt.close()

# 2. Plot pixel intensity histograms for each class
print("\n2. Plotting pixel intensity histograms...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Pixel Intensity Histograms by Class', fontsize=14)

# Normal class histogram
normal_pixels = train_images[train_labels == 0].flatten()
axes[0].hist(normal_pixels, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Normal Class')
axes[0].set_xlabel('Pixel Intensity')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Pneumonia class histogram
pneumonia_pixels = train_images[train_labels == 1].flatten()
axes[1].hist(pneumonia_pixels, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1].set_title('Pneumonia Class')
axes[1].set_xlabel('Pixel Intensity')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pixel_histograms.png'), dpi=150, bbox_inches='tight')
print(f"✓ Histograms saved to '{plots_dir}/pixel_histograms.png'")
plt.close()

# 3. Compute and report class distributions
print("\n3. Computing class distributions...")
print("\n" + "-" * 60)
print("Class Distribution:")
print("-" * 60)

def compute_distribution(labels, split_name):
    """Compute class distribution for a split."""
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

train_dist = compute_distribution(train_labels, 'Train')
val_dist = compute_distribution(val_labels, 'Validation')
test_dist = compute_distribution(test_labels, 'Test')

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

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Class Distribution Across Splits', fontsize=14)

splits = ['Train', 'Validation', 'Test']
distributions = [train_dist, val_dist, test_dist]

for ax, split_name, dist in zip(axes, splits, distributions):
    classes = ['Normal (0)', 'Pneumonia (1)']
    counts = [dist['class_0_count'], dist['class_1_count']]
    colors = ['blue', 'red']
    
    bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title(f'{split_name} Set\n(Total: {dist["total"]})')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/dist["total"]*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
print(f"✓ Class distribution plot saved to '{plots_dir}/class_distribution.png'")
plt.close()

# 4. Calculate class weights
print("\n4. Calculating class weights for weighted loss...")
total_samples = train_dist['total']
class_0_samples = train_dist['class_0_count']
class_1_samples = train_dist['class_1_count']

# Class weights formula: weight = total_samples / (num_classes * class_samples)
weight_class_0 = total_samples / (2 * class_0_samples)
weight_class_1 = total_samples / (2 * class_1_samples)

print("\n" + "-" * 60)
print("Class Weights (for Weighted Binary Cross-Entropy):")
print("-" * 60)
print(f"  Weight for Normal (0): {weight_class_0:.4f}")
print(f"  Weight for Pneumonia (1): {weight_class_1:.4f}")
print(f"\n  Formula: weight = total_samples / (2 * class_samples)")

# Save class weights
class_weights = {
    'weight_class_0': weight_class_0,
    'weight_class_1': weight_class_1,
    'train_distribution': train_dist,
    'val_distribution': val_dist,
    'test_distribution': test_dist
}

np.savez(
    os.path.join(save_dir, 'class_weights.npz'),
    weight_class_0=weight_class_0,
    weight_class_1=weight_class_1
)

print(f"\n✓ Class weights saved to '{save_dir}/class_weights.npz'")

# Summary statistics
print("\n" + "-" * 60)
print("EDA Summary:")
print("-" * 60)
print(f"Normal pixel intensity - Mean: {normal_pixels.mean():.2f}, Std: {normal_pixels.std():.2f}")
print(f"Pneumonia pixel intensity - Mean: {pneumonia_pixels.mean():.2f}, Std: {pneumonia_pixels.std():.2f}")
print(f"\nClass imbalance ratio (Pneumonia/Normal): {class_1_samples/class_0_samples:.3f}")

print("\n✓ Step 1.4 completed: Visualization and EDA done")
print(f"\n✓ All plots saved to '{plots_dir}/' directory")
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE!")
print("=" * 60)

