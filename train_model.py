"""
Complete Training Script for PneumoniaMNIST Classification
This script runs the full training pipeline with all components.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

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

print("=" * 70)
print("PNEUMONIAMNIST BINARY CLASSIFICATION - FULL TRAINING")
print("=" * 70)

# ============================================================================
# Load Data and Initialize
# ============================================================================

print("\n[1] Loading data and initializing model...")

# Load normalized data
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
print(f"  Class weights: Normal={weight_class_0:.4f}, Pneumonia={weight_class_1:.4f}")

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

print(f"\n  Hyperparameters:")
print(f"    Batch size: {BATCH_SIZE}")
print(f"    Learning rate: {LEARNING_RATE}")
print(f"    Max epochs: {MAX_EPOCHS}")
print(f"    Early stopping patience: {EARLY_STOPPING_PATIENCE}")

# Initialize model
model = PneumoniaCNN()
optimizer = SGD(learning_rate=LEARNING_RATE)

# ============================================================================
# Helper Functions
# ============================================================================

def create_batches(images, labels, batch_size, shuffle=True):
    """Create batches from dataset."""
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

def cnn_backward(model, grad_output, y_pred):
    """Complete backward pass through the CNN."""
    # Reshape gradient to match output shape
    grad_output = grad_output.reshape(-1, 1)  # (N, 1)
    
    # Gradient through sigmoid
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

def train_epoch(model, train_images, train_labels, batch_size, class_weights, learning_rate):
    """Train for one epoch."""
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
        cnn_backward(model, grad_loss, y_pred)
        
        # Update parameters
        model.dense1.update_parameters(learning_rate)
        model.dense2.update_parameters(learning_rate)
        
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, images, labels, class_weights):
    """Evaluate model on a dataset."""
    # Forward pass (no dropout)
    predictions = model.forward(images, training=False)
    predictions_flat = predictions.flatten()
    
    # Compute loss
    loss, _ = weighted_binary_cross_entropy(labels, predictions_flat, class_weights)
    
    # Binary predictions (threshold = 0.5)
    predictions_binary = (predictions_flat >= 0.5).astype(np.int32)
    
    return loss, predictions_flat, predictions_binary

def compute_metrics(y_true, y_pred_binary, y_pred_proba):
    """Compute classification metrics."""
    # Confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # AUC using sklearn
    try:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        # Fallback calculation
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_labels = y_true[sorted_indices]
        n_positive = np.sum(y_true == 1)
        n_negative = np.sum(y_true == 0)
        if n_positive > 0 and n_negative > 0:
            auc_score = 0.0
            for i, label in enumerate(sorted_labels):
                if label == 1:
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

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.stop = False
    
    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True
            return False
    
    def get_best_epoch(self):
        return self.best_epoch

# ============================================================================
# Training Loop
# ============================================================================

print("\n[2] Starting training...")

# Initialize early stopping
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)

# Storage for history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': [],
    'val_specificity': []
}

# Best model storage (we'll save the best model's state)
best_model_state = None
best_val_loss = float('inf')
best_epoch = 0

# Training loop
for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    
    # Train for one epoch
    train_loss = train_epoch(model, train_images, train_labels, BATCH_SIZE, 
                            class_weights, LEARNING_RATE)
    
    # Evaluate on validation set
    val_loss, val_pred_proba, val_pred_binary = evaluate(model, val_images, 
                                                         val_labels, class_weights)
    val_metrics = compute_metrics(val_labels, val_pred_binary, val_pred_proba)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_specificity'].append(val_metrics['specificity'])
    
    # Print progress
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f} | "
          f"Precision: {val_metrics['precision']:.4f} | "
          f"Recall: {val_metrics['recall']:.4f} | "
          f"F1: {val_metrics['f1']:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # Save model state (weights and biases)
        best_model_state = {
            'dense1_weights': model.dense1.weights.copy(),
            'dense1_bias': model.dense1.bias.copy(),
            'dense2_weights': model.dense2.weights.copy(),
            'dense2_bias': model.dense2.bias.copy()
        }
    
    # Check early stopping
    if early_stopping(val_loss, epoch):
        print(f"\n  Early stopping triggered at epoch {epoch + 1}")
        print(f"  Best epoch: {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
        break

# Restore best model
if best_model_state is not None:
    model.dense1.weights = best_model_state['dense1_weights']
    model.dense1.bias = best_model_state['dense1_bias']
    model.dense2.weights = best_model_state['dense2_weights']
    model.dense2.bias = best_model_state['dense2_bias']
    print(f"\n  Restored best model from epoch {best_epoch + 1}")

print("\n[3] Training completed!")

# ============================================================================
# Final Evaluation
# ============================================================================

print("\n[4] Final evaluation on validation and test sets...")

# Evaluate on validation set
val_loss, val_pred_proba, val_pred_binary = evaluate(model, val_images, 
                                                     val_labels, class_weights)
val_metrics = compute_metrics(val_labels, val_pred_binary, val_pred_proba)

# Evaluate on test set
test_loss, test_pred_proba, test_pred_binary = evaluate(model, test_images, 
                                                        test_labels, class_weights)
test_metrics = compute_metrics(test_labels, test_pred_binary, test_pred_proba)

print("\nValidation Set Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
print(f"  Precision: {val_metrics['precision']:.4f}")
print(f"  Recall: {val_metrics['recall']:.4f}")
print(f"  F1 Score: {val_metrics['f1']:.4f}")
print(f"  Specificity: {val_metrics['specificity']:.4f}")
print(f"  AUC: {val_metrics['auc']:.4f}")
print(f"  Confusion Matrix: TP={val_metrics['TP']}, FP={val_metrics['FP']}, "
      f"TN={val_metrics['TN']}, FN={val_metrics['FN']}")

print("\nTest Set Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1 Score: {test_metrics['f1']:.4f}")
print(f"  Specificity: {test_metrics['specificity']:.4f}")
print(f"  AUC: {test_metrics['auc']:.4f}")
print(f"  Confusion Matrix: TP={test_metrics['TP']}, FP={test_metrics['FP']}, "
      f"TN={test_metrics['TN']}, FN={test_metrics['FN']}")

# ============================================================================
# Save Results
# ============================================================================

print("\n[5] Saving results...")

# Create results directory
results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)

# Save history
np.savez(os.path.join(results_dir, 'training_history.npz'), **history)

# Save final metrics
final_metrics = {
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'total_epochs': len(history['train_loss'])
}
np.savez(os.path.join(results_dir, 'final_metrics.npz'), **final_metrics)

print(f"  Results saved to '{results_dir}/'")

# ============================================================================
# Create Plots
# ============================================================================

print("\n[6] Creating plots...")

# Plot 1: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
if best_epoch < len(history['val_loss']):
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/training_loss.png")
plt.close()

# Plot 2: Validation Metrics Over Epochs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Validation Metrics Over Epochs', fontsize=16, fontweight='bold')

axes[0, 0].plot(history['val_accuracy'], label='Accuracy', linewidth=2, color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(history['val_precision'], label='Precision', linewidth=2, color='green')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2, color='orange')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_title('Recall')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(history['val_f1'], label='F1 Score', linewidth=2, color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

if best_epoch < len(history['val_accuracy']):
    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'validation_metrics.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/validation_metrics.png")
plt.close()

# Plot 3: Confusion Matrix for Test Set
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_labels, test_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/confusion_matrix_test.png")
plt.close()

# Plot 4: ROC Curve for Test Set
fpr, tpr, thresholds = roc_curve(test_labels, test_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve_test.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/roc_curve_test.png")
plt.close()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nSummary:")
print(f"  Total epochs trained: {len(history['train_loss'])}")
print(f"  Best epoch: {best_epoch + 1}")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Test F1 score: {test_metrics['f1']:.4f}")
print(f"  Test AUC: {test_metrics['auc']:.4f}")
print(f"\nAll results and plots saved to '{results_dir}/' directory")

