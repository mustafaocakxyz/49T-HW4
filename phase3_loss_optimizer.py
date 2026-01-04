"""
Phase 3: Loss Function & Optimizer Implementation
This script implements the weighted binary cross-entropy loss and SGD optimizer.
"""

import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PHASE 3: LOSS FUNCTION & OPTIMIZER IMPLEMENTATION")
print("=" * 60)

# Load class weights
weights_data = np.load('chatgpt data normalized/class_weights.npz')
weight_class_0 = weights_data['weight_class_0']
weight_class_1 = weights_data['weight_class_1']

print(f"\nLoaded class weights:")
print(f"  Weight for Normal (0): {weight_class_0:.4f}")
print(f"  Weight for Pneumonia (1): {weight_class_1:.4f}")

# ============================================================================
# Step 3.1: Weighted Binary Cross-Entropy Loss
# ============================================================================

print("\n[Step 3.1] Implementing Weighted Binary Cross-Entropy loss...")

def weighted_binary_cross_entropy(y_true, y_pred, class_weights):
    """
    Weighted Binary Cross-Entropy loss.
    
    Args:
        y_true: true labels (N, 1) or (N,) - values 0 or 1
        y_pred: predicted probabilities (N, 1) or (N,) - values in [0, 1]
        class_weights: tuple (weight_0, weight_1) - weights for each class
    
    Returns:
        loss: scalar - average loss
        grad: gradient w.r.t. y_pred (N, 1) or (N,)
    """
    # Ensure inputs are numpy arrays and correct shape
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    # Handle 1D arrays
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    
    # Clip predictions to avoid log(0) - use larger epsilon for stability
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    weight_0, weight_1 = class_weights
    
    # Compute weighted loss
    # Loss = -[w0 * y * log(p) + w1 * (1-y) * log(1-p)]
    loss_per_sample = -(weight_0 * y_true * np.log(y_pred) + 
                        weight_1 * (1 - y_true) * np.log(1 - y_pred))
    
    # Average loss
    loss = np.mean(loss_per_sample)
    
    # Compute gradient w.r.t. y_pred
    # dL/dp = -[w0 * y / p - w1 * (1-y) / (1-p)]
    grad = -(weight_0 * y_true / y_pred - weight_1 * (1 - y_true) / (1 - y_pred))
    
    # Average gradient (since we averaged the loss)
    grad = grad / y_pred.shape[0]
    
    # Return in same shape as input
    if y_true.shape[1] == 1 and y_true.shape[0] == 1:
        return loss, grad.flatten()
    elif y_true.shape[1] == 1:
        return loss, grad.flatten()
    return loss, grad

# Test weighted binary cross-entropy
print("\nTesting Weighted Binary Cross-Entropy loss...")

# Test case 1: Perfect predictions
y_true_1 = np.array([0, 1, 0, 1], dtype=np.float32)
y_pred_1 = np.array([0.01, 0.99, 0.02, 0.98], dtype=np.float32)
loss_1, grad_1 = weighted_binary_cross_entropy(y_true_1, y_pred_1, (weight_class_0, weight_class_1))
print(f"  Test 1 - Perfect predictions:")
print(f"    Loss: {loss_1:.4f} (should be low)")
print(f"    Gradient shape: {grad_1.shape}")

# Test case 2: Poor predictions
y_true_2 = np.array([0, 1], dtype=np.float32)
y_pred_2 = np.array([0.9, 0.1], dtype=np.float32)  # Wrong predictions
loss_2, grad_2 = weighted_binary_cross_entropy(y_true_2, y_pred_2, (weight_class_0, weight_class_1))
print(f"  Test 2 - Poor predictions:")
print(f"    Loss: {loss_2:.4f} (should be high)")
print(f"    Gradient shape: {grad_2.shape}")

# Test case 3: Check gradient sign
y_true_3 = np.array([1.0], dtype=np.float32)
y_pred_3_low = np.array([0.1], dtype=np.float32)  # Under-predicted
y_pred_3_high = np.array([0.9], dtype=np.float32)  # Over-predicted
_, grad_3_low = weighted_binary_cross_entropy(y_true_3, y_pred_3_low, (weight_class_0, weight_class_1))
_, grad_3_high = weighted_binary_cross_entropy(y_true_3, y_pred_3_high, (weight_class_0, weight_class_1))
print(f"  Test 3 - Gradient direction:")
print(f"    When y_true=1, y_pred=0.1: grad = {grad_3_low[0]:.4f} (should be negative to increase prediction)")
print(f"    When y_true=1, y_pred=0.9: grad = {grad_3_high[0]:.4f} (should be less negative)")

# Verify loss increases with worse predictions
assert loss_2 > loss_1, "Loss should be higher for worse predictions"
print(f"  ✓ Loss correctly higher for worse predictions")

print("\n✓ Step 3.1 completed: Weighted Binary Cross-Entropy loss implemented and tested")

# ============================================================================
# Step 3.2: SGD Optimizer
# ============================================================================

print("\n[Step 3.2] Implementing SGD optimizer...")

class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, parameters, gradients):
        """
        Update parameters using SGD.
        
        Args:
            parameters: list of parameter arrays to update
            gradients: list of gradient arrays (same length as parameters)
        
        Returns:
            updated_parameters: list of updated parameter arrays
        """
        updated_params = []
        for param, grad in zip(parameters, gradients):
            # SGD update: θ = θ - learning_rate * gradient
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)
        return updated_params

# Test SGD optimizer
print("\nTesting SGD optimizer...")

# Create test parameters and gradients
test_weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
test_bias = np.array([0.5, 1.5], dtype=np.float32)
test_grad_weights = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
test_grad_bias = np.array([0.05, 0.15], dtype=np.float32)

print(f"  Initial weights:\n{test_weights}")
print(f"  Initial bias: {test_bias}")
print(f"  Weight gradients:\n{test_grad_weights}")
print(f"  Bias gradients: {test_grad_bias}")

# Test with different learning rates
for lr in [0.01, 0.1, 1.0]:
    sgd = SGD(learning_rate=lr)
    updated_weights, updated_bias = sgd.update([test_weights, test_bias], 
                                               [test_grad_weights, test_grad_bias])
    print(f"\n  Learning rate: {lr}")
    print(f"  Updated weights:\n{updated_weights}")
    print(f"  Updated bias: {updated_bias}")
    print(f"  Expected weights:\n{test_weights - lr * test_grad_weights}")
    print(f"  Match: {'✓' if np.allclose(updated_weights, test_weights - lr * test_grad_weights) else '✗'}")

# Verify update rule
sgd = SGD(learning_rate=0.01)
updated_params = sgd.update([test_weights], [test_grad_weights])
expected = test_weights - 0.01 * test_grad_weights
assert np.allclose(updated_params[0], expected), "SGD update rule incorrect"
print(f"\n  ✓ SGD update rule verified: θ = θ - lr * grad")

print("\n✓ Step 3.2 completed: SGD optimizer implemented and tested")
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE!")
print("=" * 60)
print("\nLoss Function & Optimizer Summary:")
print(f"  - Weighted Binary Cross-Entropy loss: ✓")
print(f"  - Class weights: Normal={weight_class_0:.4f}, Pneumonia={weight_class_1:.4f}")
print(f"  - SGD optimizer: ✓")
print(f"  - Default learning rate: 0.01 (can be tuned)")

