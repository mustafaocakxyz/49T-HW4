"""
Complete Training Script for PneumoniaMNIST Classification
Standalone version with all necessary functions included.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("PNEUMONIAMNIST BINARY CLASSIFICATION - FULL TRAINING")
print("=" * 70)

# ============================================================================
# Core Functions (included directly to avoid slow imports)
# ============================================================================

def get_emboss_kernel():
    """Return 3x3 Emboss kernel."""
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

def get_sobel_kernel():
    """Return 3x3 Sobel kernel."""
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

def conv2d(input_data, kernel, stride=1, padding=0):
    """Convolution operation."""
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    K, _ = kernel.shape
    H_out = (H - K + 2 * padding) // stride + 1
    W_out = (W - K + 2 * padding) // stride + 1
    
    output = np.zeros((N, H_out, W_out), dtype=np.float32)
    
    if padding > 0:
        padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        padded = input_data
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + K
                w_end = w_start + K
                region = padded[n, h_start:h_end, w_start:w_end]
                output[n, i, j] = np.sum(region * kernel)
    
    if single_image:
        return output[0]
    return output

def max_pool2d(input_data, pool_size=2, stride=2):
    """Max pooling operation."""
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((N, H_out, W_out), dtype=np.float32)
    indices = np.zeros((N, H_out, W_out, 2), dtype=np.int32)
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                region = input_data[n, h_start:h_end, w_start:w_end]
                max_val = np.max(region)
                max_pos = np.unravel_index(np.argmax(region), region.shape)
                output[n, i, j] = max_val
                indices[n, i, j, 0] = h_start + max_pos[0]
                indices[n, i, j, 1] = w_start + max_pos[1]
    
    if single_image:
        return output[0], indices[0]
    return output, indices

def flatten(input_data):
    """Flatten input data."""
    original_shape = input_data.shape
    if input_data.ndim == 2:
        flattened = input_data.flatten()
    else:
        N = input_data.shape[0]
        flattened = input_data.reshape(N, -1)
    return flattened, original_shape

def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative."""
    return (x > 0).astype(np.float32)

def sigmoid(x):
    """Sigmoid activation."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """Sigmoid derivative."""
    s = sigmoid(x) if np.any(x < 0) or np.any(x > 1) else x
    return s * (1 - s)

class DenseLayer:
    """Dense layer."""
    def __init__(self, input_size, output_size, initialization='xavier'):
        self.input_size = input_size
        self.output_size = output_size
        if initialization == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        elif initialization == 'he':
            std = np.sqrt(2.0 / input_size)
            self.weights = np.random.normal(0, std, (input_size, output_size)).astype(np.float32)
        else:
            self.weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
        self.bias = np.zeros(output_size, dtype=np.float32)
        self.last_input = None
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output):
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights.T)
    
    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

class DropoutLayer:
    """Dropout layer."""
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, x, training=True):
        self.training = training
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            return x * self.mask / (1 - self.rate)
        else:
            return x
    
    def backward(self, grad_output):
        if self.training and self.mask is not None:
            return grad_output * self.mask / (1 - self.rate)
        return grad_output

class PneumoniaCNN:
    """Complete CNN model."""
    def __init__(self):
        self.emboss_kernel = get_emboss_kernel()
        self.sobel_kernel = get_sobel_kernel()
        self.dense1 = DenseLayer(25, 16, 'he')
        self.dense2 = DenseLayer(16, 1, 'xavier')
        self.dropout = DropoutLayer(0.5)
        self.conv1_output = None
        self.pool1_output = None
        self.pool1_indices = None
        self.conv2_output = None
        self.pool2_output = None
        self.pool2_indices = None
        self.flattened = None
        self.flattened_shape = None
        self.dense1_output = None
        self.relu_output = None
        self.dropout_output = None
    
    def forward(self, x, training=True):
        self.conv1_output = conv2d(x, self.emboss_kernel)
        self.pool1_output, self.pool1_indices = max_pool2d(self.conv1_output, 2, 2)
        self.conv2_output = conv2d(self.pool1_output, self.sobel_kernel)
        self.pool2_output, self.pool2_indices = max_pool2d(self.conv2_output, 2, 2)
        self.flattened, self.flattened_shape = flatten(self.pool2_output)
        self.dense1_output = self.dense1.forward(self.flattened)
        self.relu_output = relu(self.dense1_output)
        self.dropout_output = self.dropout.forward(self.relu_output, training)
        dense2_output = self.dense2.forward(self.dropout_output)
        return sigmoid(dense2_output)

def weighted_binary_cross_entropy(y_true, y_pred, class_weights):
    """Weighted Binary Cross-Entropy loss."""
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    weight_0, weight_1 = class_weights
    loss_per_sample = -(weight_0 * y_true * np.log(y_pred) + 
                        weight_1 * (1 - y_true) * np.log(1 - y_pred))
    loss = np.mean(loss_per_sample)
    
    grad = -(weight_0 * y_true / y_pred - weight_1 * (1 - y_true) / (1 - y_pred))
    grad = grad / y_pred.shape[0]
    
    if y_true.shape[1] == 1 and y_true.shape[0] == 1:
        return loss, grad.flatten()
    elif y_true.shape[1] == 1:
        return loss, grad.flatten()
    return loss, grad

def conv2d_backward(grad_output, input_data, kernel):
    """Backward pass for convolution."""
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        grad_output = grad_output[np.newaxis, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    K, _ = kernel.shape
    _, H_out, W_out = grad_output.shape
    
    grad_input = np.zeros_like(input_data)
    grad_kernel = np.zeros_like(kernel)
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for ki in range(K):
                    for kj in range(K):
                        h_idx = i + ki
                        w_idx = j + kj
                        if 0 <= h_idx < H and 0 <= w_idx < W:
                            grad_input[n, h_idx, w_idx] += grad_output[n, i, j] * kernel[ki, kj]
                
                h_start = i
                w_start = j
                h_end = h_start + K
                w_end = w_start + K
                if h_end <= H and w_end <= W:
                    region = input_data[n, h_start:h_end, w_start:w_end]
                    grad_kernel += grad_output[n, i, j] * region
    
    grad_kernel = grad_kernel / N
    
    if single_image:
        return grad_input[0], grad_kernel
    return grad_input, grad_kernel

def max_pool2d_backward(grad_output, input_data, indices):
    """Backward pass for max pooling."""
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        grad_output = grad_output[np.newaxis, :, :]
        indices = indices[np.newaxis, :, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    _, H_out, W_out = grad_output.shape
    
    grad_input = np.zeros_like(input_data)
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_idx = indices[n, i, j, 0]
                w_idx = indices[n, i, j, 1]
                if 0 <= h_idx < H and 0 <= w_idx < W:
                    grad_input[n, h_idx, w_idx] += grad_output[n, i, j]
    
    if single_image:
        return grad_input[0]
    return grad_input

def flatten_backward(grad_output, original_shape):
    """Backward pass for flatten."""
    return grad_output.reshape(original_shape)

# ============================================================================
# Training Functions
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
        batches.append((images[batch_indices], labels[batch_indices]))
    return batches

def cnn_backward(model, grad_output, y_pred):
    """Complete backward pass through the CNN."""
    grad_output = grad_output.reshape(-1, 1)
    sigmoid_deriv = y_pred * (1 - y_pred)
    sigmoid_grad = grad_output * sigmoid_deriv
    
    grad_dense2_input = model.dense2.backward(sigmoid_grad)
    grad_dropout_input = model.dropout.backward(grad_dense2_input)
    grad_relu_input = grad_dropout_input * relu_derivative(model.dense1_output)
    grad_dense1_input = model.dense1.backward(grad_relu_input)
    grad_flatten_input = flatten_backward(grad_dense1_input, model.flattened_shape)
    grad_pool2_input = max_pool2d_backward(grad_flatten_input, model.pool2_output, model.pool2_indices)
    grad_conv2_input, _ = conv2d_backward(grad_pool2_input, model.pool1_output, model.sobel_kernel)
    grad_pool1_input = max_pool2d_backward(grad_conv2_input, model.conv1_output, model.pool1_indices)
    grad_conv1_input, _ = conv2d_backward(grad_pool1_input, model.conv1_output, model.emboss_kernel)

def train_epoch(model, train_images, train_labels, batch_size, class_weights, learning_rate):
    """Train for one epoch."""
    batches = create_batches(train_images, train_labels, batch_size, shuffle=True)
    total_loss = 0.0
    num_batches = 0
    
    for batch_images, batch_labels in batches:
        y_pred = model.forward(batch_images, training=True)
        loss, grad_loss = weighted_binary_cross_entropy(batch_labels, y_pred.flatten(), class_weights)
        cnn_backward(model, grad_loss, y_pred)
        model.dense1.update_parameters(learning_rate)
        model.dense2.update_parameters(learning_rate)
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, images, labels, class_weights):
    """Evaluate model on a dataset."""
    predictions = model.forward(images, training=False)
    predictions_flat = predictions.flatten()
    loss, _ = weighted_binary_cross_entropy(labels, predictions_flat, class_weights)
    predictions_binary = (predictions_flat >= 0.5).astype(np.int32)
    return loss, predictions_flat, predictions_binary

def compute_metrics(y_true, y_pred_binary, y_pred_proba):
    """Compute classification metrics."""
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        auc_score = 0.0
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'specificity': specificity, 'auc': auc_score,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN
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

# ============================================================================
# Main Training
# ============================================================================

print("\n[1] Loading data...")

data = np.load('chatgpt data normalized/normalized_data.npz')
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']

weights_data = np.load('chatgpt data normalized/class_weights.npz')
weight_class_0 = weights_data['weight_class_0']
weight_class_1 = weights_data['weight_class_1']
class_weights = (weight_class_0, weight_class_1)

print(f"  Training: {train_images.shape[0]}, Validation: {val_images.shape[0]}, Test: {test_images.shape[0]}")

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

print(f"\n[2] Initializing model...")
model = PneumoniaCNN()
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)

history = {
    'train_loss': [], 'val_loss': [], 'val_accuracy': [],
    'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_specificity': []
}

best_model_state = None
best_val_loss = float('inf')
best_epoch = 0

print(f"\n[3] Starting training (max {MAX_EPOCHS} epochs, early stopping patience={EARLY_STOPPING_PATIENCE})...")
print("=" * 70)

for epoch in range(MAX_EPOCHS):
    # Train
    train_loss = train_epoch(model, train_images, train_labels, BATCH_SIZE, class_weights, LEARNING_RATE)
    
    # Validate
    val_loss, val_pred_proba, val_pred_binary = evaluate(model, val_images, val_labels, class_weights)
    val_metrics = compute_metrics(val_labels, val_pred_binary, val_pred_proba)
    
    # Store
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_specificity'].append(val_metrics['specificity'])
    
    # Print
    print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = {
            'dense1_weights': model.dense1.weights.copy(),
            'dense1_bias': model.dense1.bias.copy(),
            'dense2_weights': model.dense2.weights.copy(),
            'dense2_bias': model.dense2.bias.copy()
        }
    
    # Early stopping
    if early_stopping(val_loss, epoch):
        print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f})")
        break

# Restore best model
if best_model_state is not None:
    model.dense1.weights = best_model_state['dense1_weights']
    model.dense1.bias = best_model_state['dense1_bias']
    model.dense2.weights = best_model_state['dense2_weights']
    model.dense2.bias = best_model_state['dense2_bias']

print("\n[4] Final evaluation...")
val_loss, val_pred_proba, val_pred_binary = evaluate(model, val_images, val_labels, class_weights)
val_metrics = compute_metrics(val_labels, val_pred_binary, val_pred_proba)

test_loss, test_pred_proba, test_pred_binary = evaluate(model, test_images, test_labels, class_weights)
test_metrics = compute_metrics(test_labels, test_pred_binary, test_pred_proba)

print(f"\nValidation: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
print(f"Test:       Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")

# Save results
print("\n[5] Saving results...")
results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)
np.savez(os.path.join(results_dir, 'training_history.npz'), **history)
np.savez(os.path.join(results_dir, 'final_metrics.npz'),
         val_metrics=val_metrics, test_metrics=test_metrics,
         best_epoch=best_epoch, best_val_loss=best_val_loss)

# Create plots
print("[6] Creating plots...")
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
if best_epoch < len(history['val_loss']):
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Validation Metrics Over Epochs', fontsize=16, fontweight='bold')
axes[0,0].plot(history['val_accuracy'], label='Accuracy', linewidth=2); axes[0,0].set_title('Accuracy'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
axes[0,1].plot(history['val_precision'], label='Precision', linewidth=2); axes[0,1].set_title('Precision'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[1,0].plot(history['val_recall'], label='Recall', linewidth=2); axes[1,0].set_title('Recall'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
axes[1,1].plot(history['val_f1'], label='F1 Score', linewidth=2); axes[1,1].set_title('F1 Score'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
if best_epoch < len(history['val_accuracy']):
    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'validation_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()

cm = confusion_matrix(test_labels, test_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.png'), dpi=150, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(test_labels, test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set'); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve_test.png'), dpi=150, bbox_inches='tight')
plt.close()

print("=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Total epochs: {len(history['train_loss'])} | Best epoch: {best_epoch+1}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")
print(f"Results saved to '{results_dir}/'")

