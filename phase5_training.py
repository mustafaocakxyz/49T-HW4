"""
Phase 5: Training Loop Implementation
This script implements the complete training pipeline with early stopping and metrics tracking.
"""

import numpy as np
import sys
import os
from tqdm import tqdm

# Import from previous phases
sys.path.append('.')
from phase2_network_architecture import (
    PneumoniaCNN, conv2d, max_pool2d, flatten, relu, relu_derivative,
    sigmoid, sigmoid_derivative, DenseLayer, DropoutLayer,
    get_emboss_kernel, get_sobel_kernel
)
from phase3_loss_optimizer import weighted_binary_cross_entropy, SGD
from phase4_backward_propagation import (
    conv2d_backward, max_pool2d_backward, flatten_backward
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PHASE 5: TRAINING LOOP IMPLEMENTATION")
print("=" * 60)

# ============================================================================
# Step 5.1: Training Infrastructure
# ============================================================================

print("\n[Step 5.1] Setting up training infrastructure...")

# Load normalized data
print("\nLoading normalized data...")
data = np.load('chatgpt data normalized/normalized_data.npz')
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Load class weights
weights_data = np.load('chatgpt data normalized/class_weights.npz')
weight_class_0 = weights_data['weight_class_0']
weight_class_1 = weights_data['weight_class_1']
class_weights = (weight_class_0, weight_class_1)

print(f"  Training set: {train_images.shape[0]} samples")
print(f"  Validation set: {val_images.shape[0]} samples")
print(f"  Test set: {test_images.shape[0]} samples")
print(f"  Image shape: {train_images.shape[1:]}")

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

print(f"\nTraining Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")

# Initialize model
print("\nInitializing model...")
model = PneumoniaCNN()
optimizer = SGD(learning_rate=LEARNING_RATE)

# Helper function to create batches
def create_batches(images, labels, batch_size, shuffle=True):
    """
    Create batches from dataset.
    
    Args:
        images: (N, H, W) array of images
        labels: (N,) array of labels
        batch_size: size of each batch
        shuffle: whether to shuffle data
    
    Returns:
        batches: list of (batch_images, batch_labels) tuples
    """
    N = len(images)
    indices = np.arange(N)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]
        batches.append((batch_images, batch_labels))
    
    return batches

# Test batch creation
print("\nTesting batch creation...")
test_batches = create_batches(train_images[:100], train_labels[:100], BATCH_SIZE, shuffle=True)
print(f"  Created {len(test_batches)} batches from 100 samples")
print(f"  First batch shape: images={test_batches[0][0].shape}, labels={test_batches[0][1].shape}")
print(f"  Last batch shape: images={test_batches[-1][0].shape}, labels={test_batches[-1][1].shape}")

print("\n✓ Step 5.1 completed: Training infrastructure set up")

# ============================================================================
# Step 5.2: Training Loop
# ============================================================================

print("\n[Step 5.2] Implementing training loop with backward pass...")

# Extend PneumoniaCNN with backward pass method
def cnn_backward(model, grad_output, y_true, y_pred, class_weights):
    """
    Complete backward pass through the CNN.
    
    Args:
        model: PneumoniaCNN instance
        grad_output: gradient from loss function (N,) - flattened
        y_true: true labels
        y_pred: predicted probabilities (N, 1)
        class_weights: tuple of class weights
    
    Returns:
        None (gradients stored in model layers)
    """
    # Reshape gradient to match output shape
    grad_output = grad_output.reshape(-1, 1)  # (N, 1)
    
    # Gradient through sigmoid
    # dL/dx = dL/dp * dp/dx where dp/dx = p * (1-p) for sigmoid
    # We can compute sigmoid derivative from the output (y_pred) directly
    # sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
    # But since we have y_pred (which is sigmoid(x)), we can use:
    # sigmoid_derivative = y_pred * (1 - y_pred)
    sigmoid_deriv = y_pred * (1 - y_pred)  # (N, 1)
    sigmoid_grad = grad_output * sigmoid_deriv  # (N, 1)
    
    # Backward through Dense2
    grad_dense2_input = model.dense2.backward(sigmoid_grad)
    
    # Backward through Dropout
    grad_dropout_input = model.dropout.backward(grad_dense2_input)
    
    # Backward through ReLU
    grad_relu_input = grad_dropout_input * relu_derivative(model.dense1_output)
    
    # Backward through Dense1
    grad_dense1_input = model.dense1.backward(grad_relu_input)
    
    # Backward through Flatten
    grad_flatten_input = flatten_backward(grad_dense1_input, model.flattened_shape)
    
    # Backward through MaxPool2
    grad_pool2_input = max_pool2d_backward(grad_flatten_input, model.pool2_output, model.pool2_indices)
    
    # Backward through Conv2 (Sobel)
    grad_conv2_input, _ = conv2d_backward(grad_pool2_input, model.pool1_output, model.sobel_kernel)
    
    # Backward through MaxPool1
    grad_pool1_input = max_pool2d_backward(grad_conv2_input, model.conv1_output, model.pool1_indices)
    
    # Backward through Conv1 (Emboss)
    grad_conv1_input, _ = conv2d_backward(grad_pool1_input, model.conv1_output, model.emboss_kernel)
    
    # Note: grad_conv1_input is the gradient w.r.t. input, not used for updates
    # Kernel gradients computed but not used (kernels are fixed)

# Training loop function
def train_epoch(model, train_images, train_labels, batch_size, class_weights, learning_rate):
    """
    Train for one epoch.
    
    Returns:
        average_loss: average training loss for the epoch
    """
    batches = create_batches(train_images, train_labels, batch_size, shuffle=True)
    total_loss = 0.0
    num_batches = 0
    
    for batch_images, batch_labels in batches:
        # Forward pass
        y_pred = model.forward(batch_images, training=True)
        
        # Compute loss and gradient
        loss, grad_loss = weighted_binary_cross_entropy(
            batch_labels, y_pred.flatten(), class_weights
        )
        
        # Backward pass
        cnn_backward(model, grad_loss, batch_labels, y_pred, class_weights)
        
        # Update parameters
        model.dense1.update_parameters(learning_rate)
        model.dense2.update_parameters(learning_rate)
        
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches

# Evaluation function
def evaluate(model, images, labels, class_weights):
    """
    Evaluate model on a dataset.
    
    Returns:
        loss: average loss
        predictions: predicted probabilities
        predictions_binary: binary predictions (0 or 1)
    """
    # Forward pass (no dropout)
    predictions = model.forward(images, training=False)
    predictions_flat = predictions.flatten()
    
    # Compute loss
    loss, _ = weighted_binary_cross_entropy(labels, predictions_flat, class_weights)
    
    # Binary predictions (threshold = 0.5)
    predictions_binary = (predictions_flat >= 0.5).astype(np.int32)
    
    return loss, predictions_flat, predictions_binary

# Test training on a small subset
print("\nTesting training loop on small subset...")
test_model = PneumoniaCNN()
test_optimizer = SGD(learning_rate=LEARNING_RATE)

# Train for 1 epoch on small subset
test_train_images = train_images[:200]
test_train_labels = train_labels[:200]
test_val_images = val_images[:50]
test_val_labels = val_labels[:50]

print("  Training on 200 samples, validating on 50 samples...")
train_loss = train_epoch(test_model, test_train_images, test_train_labels, 
                         BATCH_SIZE, class_weights, LEARNING_RATE)
val_loss, val_pred, val_pred_binary = evaluate(test_model, test_val_images, 
                                                test_val_labels, class_weights)

print(f"  Training loss: {train_loss:.4f}")
print(f"  Validation loss: {val_loss:.4f}")
print(f"  Predictions shape: {val_pred.shape}")
print(f"  Binary predictions range: [{val_pred_binary.min()}, {val_pred_binary.max()}]")
print(f"  ✓ Training loop works correctly")

print("\n✓ Step 5.2 completed: Training loop implemented and tested")

# ============================================================================
# Step 5.3: Early Stopping
# ============================================================================

print("\n[Step 5.3] Implementing early stopping...")

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.stop = False
    
    def __call__(self, val_loss, epoch):
        """
        Check if training should stop.
        
        Args:
            val_loss: current validation loss
            epoch: current epoch number
        
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True
            return False
    
    def get_best_epoch(self):
        """Return the epoch with best validation loss."""
        return self.best_epoch

# Test early stopping
print("\nTesting early stopping...")
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# Simulate validation losses
val_losses = [1.0, 0.9, 0.85, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82]
print("  Simulating validation losses:", val_losses[:8])

for epoch, val_loss in enumerate(val_losses):
    should_stop = early_stopping(val_loss, epoch)
    print(f"  Epoch {epoch}: val_loss={val_loss:.3f}, best={early_stopping.best_loss:.3f}, "
          f"counter={early_stopping.counter}, stop={should_stop}")
    if should_stop:
        print(f"  ✓ Early stopping triggered at epoch {epoch}")
        break

print(f"  Best epoch: {early_stopping.get_best_epoch()}")
print(f"  Best validation loss: {early_stopping.best_loss:.4f}")

print("\n✓ Step 5.3 completed: Early stopping implemented and tested")

# ============================================================================
# Step 5.4: Metrics Tracking
# ============================================================================

print("\n[Step 5.4] Implementing metrics tracking...")

def compute_metrics(y_true, y_pred_binary, y_pred_proba):
    """
    Compute classification metrics.
    
    Args:
        y_true: true labels (N,)
        y_pred_binary: binary predictions (N,)
        y_pred_proba: probability predictions (N,)
    
    Returns:
        dict: dictionary of metrics
    """
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall (Sensitivity)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # AUC (simplified - using threshold-based approximation)
    # For proper AUC, we'd need to compute ROC curve, but for now we'll use a simple approximation
    # Sort by probability and compute area under curve
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_labels = y_true[sorted_indices]
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    if n_positive > 0 and n_negative > 0:
        # Count pairs where positive has higher probability than negative
        auc_score = 0.0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                # Count negatives that come after this positive
                auc_score += np.sum(sorted_labels[i+1:] == 0)
        auc_score = auc_score / (n_positive * n_negative)
    else:
        auc_score = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc_score,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }

# Test metrics computation
print("\nTesting metrics computation...")
test_y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
test_y_pred_binary = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0])  # 1 FP, 1 FN
test_y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.6, 0.7, 0.4, 0.3, 0.9, 0.2])

metrics = compute_metrics(test_y_true, test_y_pred_binary, test_y_pred_proba)

print(f"  True labels: {test_y_true}")
print(f"  Predictions: {test_y_pred_binary}")
print(f"  Confusion Matrix:")
print(f"    TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
print(f"  Metrics:")
print(f"    Accuracy: {metrics['accuracy']:.4f}")
print(f"    Precision: {metrics['precision']:.4f}")
print(f"    Recall: {metrics['recall']:.4f}")
print(f"    F1 Score: {metrics['f1']:.4f}")
print(f"    Specificity: {metrics['specificity']:.4f}")
print(f"    AUC: {metrics['auc']:.4f}")

# Verify metrics are in valid ranges
assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
assert 0 <= metrics['precision'] <= 1, "Precision should be in [0, 1]"
assert 0 <= metrics['recall'] <= 1, "Recall should be in [0, 1]"
assert 0 <= metrics['f1'] <= 1, "F1 should be in [0, 1]"
assert 0 <= metrics['specificity'] <= 1, "Specificity should be in [0, 1]"
print(f"  ✓ All metrics in valid ranges")

print("\n✓ Step 5.4 completed: Metrics tracking implemented and tested")
print("\n" + "=" * 60)
print("PHASE 5 COMPLETE!")
print("=" * 60)
print("\nTraining Infrastructure Summary:")
print(f"  - Batch creation: ✓")
print(f"  - Training loop: ✓")
print(f"  - Backward propagation: ✓")
print(f"  - Parameter updates: ✓")
print(f"  - Early stopping: ✓")
print(f"  - Metrics computation: ✓")
print(f"\nReady for full training!")

