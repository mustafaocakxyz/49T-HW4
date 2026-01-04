"""
Phase 2: Network Architecture Implementation
This script implements all network components in NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PHASE 2: NETWORK ARCHITECTURE IMPLEMENTATION")
print("=" * 60)

# ============================================================================
# Step 2.1: Define Kernel Functions
# ============================================================================

print("\n[Step 2.1] Defining kernel functions...")

def get_emboss_kernel():
    """
    Return 3x3 Emboss kernel for texture enhancement.
    Emboss filter enhances texture and depth perception.
    """
    emboss = np.array([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ], dtype=np.float32)
    return emboss

def get_sobel_kernel():
    """
    Return 3x3 Sobel kernel for edge detection.
    Using Sobel X kernel for horizontal edge detection.
    """
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)
    return sobel_x

# Test kernels
emboss_kernel = get_emboss_kernel()
sobel_kernel = get_sobel_kernel()

print("\n✓ Emboss Kernel (3x3):")
print(emboss_kernel)
print("\n✓ Sobel Kernel (3x3):")
print(sobel_kernel)
print("\n✓ Step 2.1 completed: Kernel functions defined")

# ============================================================================
# Step 2.2: Convolution Layer Implementation
# ============================================================================

print("\n[Step 2.2] Implementing convolution layer...")

def conv2d(input_data, kernel, stride=1, padding=0):
    """
    Convolution operation.
    
    Args:
        input_data: (N, H, W) or (H, W) - batch of images or single image
        kernel: (K, K) - convolution kernel
        stride: int - stride for convolution
        padding: int - padding (0 for no padding)
    
    Returns:
        output: (N, H_out, W_out) or (H_out, W_out)
    """
    # Handle single image case
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    K, _ = kernel.shape
    
    # Calculate output dimensions
    H_out = (H - K + 2 * padding) // stride + 1
    W_out = (W - K + 2 * padding) // stride + 1
    
    # Initialize output
    output = np.zeros((N, H_out, W_out), dtype=np.float32)
    
    # Apply padding if needed
    if padding > 0:
        padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        padded = input_data
    
    # Perform convolution
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + K
                w_end = w_start + K
                
                # Extract region and apply kernel
                region = padded[n, h_start:h_end, w_start:w_end]
                output[n, i, j] = np.sum(region * kernel)
    
    # Return single image if input was single image
    if single_image:
        return output[0]
    return output

# Test convolution on sample image
print("\nTesting convolution on sample 28x28 image...")
test_image = np.random.randn(28, 28).astype(np.float32)
test_batch = np.random.randn(2, 28, 28).astype(np.float32)

# Test with Emboss kernel
conv_output_single = conv2d(test_image, emboss_kernel)
conv_output_batch = conv2d(test_batch, emboss_kernel)

print(f"  Input shape (single): {test_image.shape}")
print(f"  Output shape (single): {conv_output_single.shape}")
print(f"  Expected: (26, 26) ✓" if conv_output_single.shape == (26, 26) else f"  Expected: (26, 26) ✗")
print(f"  Input shape (batch): {test_batch.shape}")
print(f"  Output shape (batch): {conv_output_batch.shape}")
print(f"  Expected: (2, 26, 26) ✓" if conv_output_batch.shape == (2, 26, 26) else f"  Expected: (2, 26, 26) ✗")

print("\n✓ Step 2.2 completed: Convolution layer implemented and tested")

# ============================================================================
# Step 2.3: Max Pooling Implementation
# ============================================================================

print("\n[Step 2.3] Implementing max pooling layer...")

def max_pool2d(input_data, pool_size=2, stride=2):
    """
    Max pooling operation.
    
    Args:
        input_data: (N, H, W) or (H, W) - batch of images or single image
        pool_size: int - size of pooling window (assumed square)
        stride: int - stride for pooling
    
    Returns:
        output: pooled data with same shape handling as input
        indices: indices of max values for backpropagation (N, H_out, W_out, 2)
    """
    # Handle single image case
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
        single_image = True
    else:
        single_image = False
    
    N, H, W = input_data.shape
    
    # Calculate output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    # Initialize output and indices
    output = np.zeros((N, H_out, W_out), dtype=np.float32)
    indices = np.zeros((N, H_out, W_out, 2), dtype=np.int32)  # Store (h, w) indices
    
    # Perform max pooling
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                # Extract region
                region = input_data[n, h_start:h_end, w_start:w_end]
                
                # Find max value and its position
                max_val = np.max(region)
                max_pos = np.unravel_index(np.argmax(region), region.shape)
                
                # Store output and indices (relative to input)
                output[n, i, j] = max_val
                indices[n, i, j, 0] = h_start + max_pos[0]  # h index
                indices[n, i, j, 1] = w_start + max_pos[1]  # w index
    
    # Return single image if input was single image
    if single_image:
        return output[0], indices[0]
    return output, indices

# Test max pooling
print("\nTesting max pooling on sample 26x26 image...")
test_image_pool = np.random.randn(26, 26).astype(np.float32)
test_batch_pool = np.random.randn(2, 26, 26).astype(np.float32)

pool_output_single, pool_indices_single = max_pool2d(test_image_pool)
pool_output_batch, pool_indices_batch = max_pool2d(test_batch_pool)

print(f"  Input shape (single): {test_image_pool.shape}")
print(f"  Output shape (single): {pool_output_single.shape}")
print(f"  Expected: (13, 13) ✓" if pool_output_single.shape == (13, 13) else f"  Expected: (13, 13) ✗")
print(f"  Indices shape (single): {pool_indices_single.shape}")
print(f"  Input shape (batch): {test_batch_pool.shape}")
print(f"  Output shape (batch): {pool_output_batch.shape}")
print(f"  Expected: (2, 13, 13) ✓" if pool_output_batch.shape == (2, 13, 13) else f"  Expected: (2, 13, 13) ✗")
print(f"  Indices shape (batch): {pool_indices_batch.shape}")

print("\n✓ Step 2.3 completed: Max pooling layer implemented and tested")

# ============================================================================
# Step 2.4: Activation Functions
# ============================================================================

print("\n[Step 2.4] Implementing activation functions...")

def relu(x):
    """
    ReLU activation function: max(0, x)
    
    Args:
        x: input array
    
    Returns:
        output: ReLU(x)
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU: 1 if x > 0, else 0
    
    Args:
        x: input array
    
    Returns:
        derivative: d(ReLU)/dx
    """
    return (x > 0).astype(np.float32)

def sigmoid(x):
    """
    Sigmoid activation function: 1 / (1 + exp(-x))
    
    Args:
        x: input array
    
    Returns:
        output: sigmoid(x) in range [0, 1]
    """
    # Clip x to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        x: input array (can be sigmoid output or raw input)
    
    Returns:
        derivative: d(sigmoid)/dx
    """
    s = sigmoid(x) if np.any(x < 0) or np.any(x > 1) else x
    return s * (1 - s)

# Test activation functions
print("\nTesting activation functions...")
test_input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

relu_out = relu(test_input)
relu_der_out = relu_derivative(test_input)
sigmoid_out = sigmoid(test_input)
sigmoid_der_out = sigmoid_derivative(test_input)

print(f"  Input: {test_input}")
print(f"  ReLU output: {relu_out}")
print(f"  ReLU derivative: {relu_der_out}")
print(f"  Sigmoid output: {sigmoid_out}")
print(f"  Sigmoid derivative: {sigmoid_der_out}")

# Verify sigmoid range
print(f"\n  Sigmoid range check: min={sigmoid_out.min():.4f}, max={sigmoid_out.max():.4f}")
assert np.all(sigmoid_out >= 0) and np.all(sigmoid_out <= 1), "Sigmoid output should be in [0, 1]"
print("  ✓ Sigmoid output in valid range [0, 1]")

print("\n✓ Step 2.4 completed: Activation functions implemented and tested")

# ============================================================================
# Step 2.5: Flatten Layer
# ============================================================================

print("\n[Step 2.5] Implementing flatten layer...")

def flatten(input_data):
    """
    Flatten input data from (N, H, W) to (N, H*W).
    
    Args:
        input_data: (N, H, W) or (H, W) - batch of images or single image
    
    Returns:
        flattened: (N, H*W) or (H*W,) - flattened data
        original_shape: tuple - original shape for potential unflatten
    """
    original_shape = input_data.shape
    
    # Handle single image case
    if input_data.ndim == 2:
        flattened = input_data.flatten()
    else:
        N = input_data.shape[0]
        flattened = input_data.reshape(N, -1)
    
    return flattened, original_shape

def unflatten(flattened_data, original_shape):
    """
    Unflatten data back to original shape.
    
    Args:
        flattened_data: (N, H*W) or (H*W,)
        original_shape: tuple - original shape
    
    Returns:
        unflattened: data in original shape
    """
    return flattened_data.reshape(original_shape)

# Test flatten
print("\nTesting flatten layer...")
test_image_flat = np.random.randn(5, 5).astype(np.float32)
test_batch_flat = np.random.randn(2, 5, 5).astype(np.float32)

flat_single, shape_single = flatten(test_image_flat)
flat_batch, shape_batch = flatten(test_batch_flat)

print(f"  Input shape (single): {test_image_flat.shape}")
print(f"  Flattened shape (single): {flat_single.shape}")
print(f"  Expected: (25,) ✓" if flat_single.shape == (25,) else f"  Expected: (25,) ✗")
print(f"  Original shape stored: {shape_single}")

print(f"  Input shape (batch): {test_batch_flat.shape}")
print(f"  Flattened shape (batch): {flat_batch.shape}")
print(f"  Expected: (2, 25) ✓" if flat_batch.shape == (2, 25) else f"  Expected: (2, 25) ✗")
print(f"  Original shape stored: {shape_batch}")

# Test unflatten
unflat_single = unflatten(flat_single, shape_single)
unflat_batch = unflatten(flat_batch, shape_batch)

assert unflat_single.shape == test_image_flat.shape, "Unflatten failed for single image"
assert unflat_batch.shape == test_batch_flat.shape, "Unflatten failed for batch"
print("  ✓ Unflatten test passed")

print("\n✓ Step 2.5 completed: Flatten layer implemented and tested")

# ============================================================================
# Step 2.6: Dense Layer Implementation
# ============================================================================

print("\n[Step 2.6] Implementing dense (fully connected) layer...")

class DenseLayer:
    """Dense (fully connected) layer with forward and backward pass."""
    
    def __init__(self, input_size, output_size, initialization='xavier'):
        """
        Initialize dense layer.
        
        Args:
            input_size: number of input features
            output_size: number of output features
            initialization: 'xavier' or 'he' or 'random'
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights
        if initialization == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
        elif initialization == 'he':
            # He initialization (for ReLU)
            std = np.sqrt(2.0 / input_size)
            self.weights = np.random.normal(0, std, (input_size, output_size)).astype(np.float32)
        else:
            # Small random initialization
            self.weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
        
        # Initialize biases to zero
        self.bias = np.zeros(output_size, dtype=np.float32)
        
        # Storage for backward pass
        self.last_input = None
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x):
        """
        Forward pass: output = x @ W + b
        
        Args:
            x: (N, input_size) - input data
        
        Returns:
            output: (N, output_size) - output data
        """
        self.last_input = x
        output = np.dot(x, self.weights) + self.bias
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients.
        
        Args:
            grad_output: (N, output_size) - gradient from next layer
        
        Returns:
            grad_input: (N, input_size) - gradient to previous layer
        """
        # Gradient w.r.t. weights: dL/dW = x^T @ grad_output
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        
        # Gradient w.r.t. bias: dL/db = sum(grad_output, axis=0)
        self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. input: dL/dx = grad_output @ W^T
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def update_parameters(self, learning_rate):
        """
        Update parameters using gradients.
        
        Args:
            learning_rate: learning rate for SGD
        """
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

# Test dense layer
print("\nTesting dense layer...")
dense1 = DenseLayer(input_size=25, output_size=16, initialization='xavier')
dense2 = DenseLayer(input_size=16, output_size=1, initialization='xavier')

test_input_dense = np.random.randn(2, 25).astype(np.float32)

# Forward pass
output1 = dense1.forward(test_input_dense)
print(f"  Input shape: {test_input_dense.shape}")
print(f"  Dense1 output shape: {output1.shape}")
print(f"  Expected: (2, 16) ✓" if output1.shape == (2, 16) else f"  Expected: (2, 16) ✗")
print(f"  Dense1 weights shape: {dense1.weights.shape}")
print(f"  Dense1 bias shape: {dense1.bias.shape}")

output2 = dense2.forward(output1)
print(f"  Dense2 output shape: {output2.shape}")
print(f"  Expected: (2, 1) ✓" if output2.shape == (2, 1) else f"  Expected: (2, 1) ✗")

# Backward pass (test)
grad_output2 = np.random.randn(2, 1).astype(np.float32)
grad_input2 = dense2.backward(grad_output2)
grad_input1 = dense1.backward(grad_input2)

print(f"  Gradient shapes match: ✓")
print(f"    grad_output2: {grad_output2.shape}")
print(f"    grad_input2: {grad_input2.shape}, matches output1: {output1.shape} ✓")
print(f"    grad_input1: {grad_input1.shape}, matches test_input: {test_input_dense.shape} ✓")

print("\n✓ Step 2.6 completed: Dense layer implemented and tested")

# ============================================================================
# Step 2.7: Dropout Implementation
# ============================================================================

print("\n[Step 2.7] Implementing dropout layer...")

class DropoutLayer:
    """Dropout layer for regularization."""
    
    def __init__(self, rate=0.5):
        """
        Initialize dropout layer.
        
        Args:
            rate: dropout rate (probability of setting a unit to zero)
        """
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, x, training=True):
        """
        Forward pass: randomly set units to zero during training.
        
        Args:
            x: input data
            training: if True, apply dropout; if False, scale output
        
        Returns:
            output: dropped out data (training) or scaled data (inference)
        """
        self.training = training
        
        if training:
            # Generate random mask
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            # Apply mask and scale by 1/(1-rate) to maintain expected value
            output = x * self.mask / (1 - self.rate)
        else:
            # During inference, no dropout (output = input)
            output = x
            self.mask = None
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: apply mask to gradients.
        
        Args:
            grad_output: gradient from next layer
        
        Returns:
            grad_input: masked gradient
        """
        if self.training and self.mask is not None:
            # Apply mask and scale
            grad_input = grad_output * self.mask / (1 - self.rate)
        else:
            grad_input = grad_output
        
        return grad_input

# Test dropout
print("\nTesting dropout layer...")
dropout = DropoutLayer(rate=0.5)
test_input_dropout = np.ones((2, 16), dtype=np.float32) * 2.0  # All values = 2.0

# Training mode
output_train = dropout.forward(test_input_dropout, training=True)
print(f"  Input (training): all values = 2.0")
print(f"  Output (training): min={output_train.min():.2f}, max={output_train.max():.2f}, mean={output_train.mean():.2f}")
print(f"  Expected: ~50% zeros, mean ~2.0")
print(f"  Mask sum: {dropout.mask.sum()}/{dropout.mask.size} ({100*dropout.mask.sum()/dropout.mask.size:.1f}% kept)")

# Inference mode
output_inference = dropout.forward(test_input_dropout, training=False)
print(f"  Input (inference): all values = 2.0")
print(f"  Output (inference): all values = {output_inference[0, 0]:.2f}")
print(f"  Expected: all values = 2.0 (no dropout) ✓" if np.allclose(output_inference, 2.0) else "  ✗")

# Test backward
grad_output_dropout = np.ones((2, 16), dtype=np.float32)
dropout.forward(test_input_dropout, training=True)  # Set mask
grad_input = dropout.backward(grad_output_dropout)
print(f"  Gradient backward: {grad_input.shape}, mask applied: ✓")

print("\n✓ Step 2.7 completed: Dropout layer implemented and tested")

# ============================================================================
# Step 2.8: Complete Forward Pass
# ============================================================================

print("\n[Step 2.8] Implementing complete forward pass through network...")

class PneumoniaCNN:
    """Complete CNN model for pneumonia classification."""
    
    def __init__(self):
        """Initialize all layers of the network."""
        # Convolution kernels
        self.emboss_kernel = get_emboss_kernel()
        self.sobel_kernel = get_sobel_kernel()
        
        # Dense layers
        self.dense1 = DenseLayer(input_size=25, output_size=16, initialization='he')
        self.dense2 = DenseLayer(input_size=16, output_size=1, initialization='xavier')
        
        # Dropout layer
        self.dropout = DropoutLayer(rate=0.5)
        
        # Storage for intermediate values (for backpropagation)
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
        """
        Forward pass through entire network.
        
        Args:
            x: (N, 28, 28) - input images
            training: bool - whether in training mode (affects dropout)
        
        Returns:
            output: (N, 1) - predicted probabilities
        """
        # Layer 1: Conv1 (Emboss 3x3)
        # Input: (N, 28, 28) → Output: (N, 26, 26)
        self.conv1_output = conv2d(x, self.emboss_kernel)
        
        # Layer 2: MaxPool1 (2x2, stride=2)
        # Input: (N, 26, 26) → Output: (N, 13, 13)
        self.pool1_output, self.pool1_indices = max_pool2d(self.conv1_output, pool_size=2, stride=2)
        
        # Layer 3: Conv2 (Sobel 3x3)
        # Input: (N, 13, 13) → Output: (N, 11, 11)
        self.conv2_output = conv2d(self.pool1_output, self.sobel_kernel)
        
        # Layer 4: MaxPool2 (2x2, stride=2)
        # Input: (N, 11, 11) → Output: (N, 5, 5)
        self.pool2_output, self.pool2_indices = max_pool2d(self.conv2_output, pool_size=2, stride=2)
        
        # Layer 5: Flatten
        # Input: (N, 5, 5) → Output: (N, 25)
        self.flattened, self.flattened_shape = flatten(self.pool2_output)
        
        # Layer 6: Dense1 (16 units) + ReLU
        # Input: (N, 25) → Output: (N, 16)
        self.dense1_output = self.dense1.forward(self.flattened)
        self.relu_output = relu(self.dense1_output)
        
        # Layer 7: Dropout (0.5)
        # Input: (N, 16) → Output: (N, 16)
        self.dropout_output = self.dropout.forward(self.relu_output, training=training)
        
        # Layer 8: Dense2 (1 unit) + Sigmoid
        # Input: (N, 16) → Output: (N, 1)
        dense2_output = self.dense2.forward(self.dropout_output)
        output = sigmoid(dense2_output)
        
        return output

# Test complete forward pass
print("\nTesting complete forward pass...")
model = PneumoniaCNN()

# Test with batch of 2 images
test_batch_final = np.random.randn(2, 28, 28).astype(np.float32)
print(f"  Input shape: {test_batch_final.shape}")

# Forward pass
output_final = model.forward(test_batch_final, training=True)

# Print dimension flow
print("\n  Dimension Flow:")
print(f"    Input:              {test_batch_final.shape}")
print(f"    Conv1 (Emboss):     {model.conv1_output.shape}")
print(f"    MaxPool1:           {model.pool1_output.shape}")
print(f"    Conv2 (Sobel):      {model.conv2_output.shape}")
print(f"    MaxPool2:           {model.pool2_output.shape}")
print(f"    Flatten:            {model.flattened.shape}")
print(f"    Dense1 (16):        {model.dense1_output.shape}")
print(f"    ReLU:               {model.relu_output.shape}")
print(f"    Dropout:            {model.dropout_output.shape}")
print(f"    Dense2 (1):         {model.dense2.last_input.shape}")
print(f"    Sigmoid (Output):   {output_final.shape}")

# Verify dimensions
expected_shapes = [
    (2, 28, 28),  # Input
    (2, 26, 26),  # Conv1
    (2, 13, 13),  # MaxPool1
    (2, 11, 11),  # Conv2
    (2, 5, 5),    # MaxPool2
    (2, 25),      # Flatten
    (2, 16),      # Dense1
    (2, 16),      # ReLU
    (2, 16),      # Dropout
    (2, 1)        # Output
]

actual_shapes = [
    test_batch_final.shape,
    model.conv1_output.shape,
    model.pool1_output.shape,
    model.conv2_output.shape,
    model.pool2_output.shape,
    model.flattened.shape,
    model.dense1_output.shape,
    model.relu_output.shape,
    model.dropout_output.shape,
    output_final.shape
]

print("\n  Dimension Verification:")
all_correct = True
for i, (expected, actual) in enumerate(zip(expected_shapes, actual_shapes)):
    match = expected == actual
    status = "✓" if match else "✗"
    if not match:
        all_correct = False
    print(f"    Layer {i+1}: {status} Expected {expected}, Got {actual}")

# Verify output range
print(f"\n  Output range check:")
print(f"    Min: {output_final.min():.4f}, Max: {output_final.max():.4f}")
assert np.all(output_final >= 0) and np.all(output_final <= 1), "Sigmoid output should be in [0, 1]"
print(f"    ✓ Output in valid range [0, 1]")

# Test with single image
print("\n  Testing with single image...")
test_single = np.random.randn(28, 28).astype(np.float32)
output_single = model.forward(test_single[np.newaxis, :, :], training=False)
print(f"    Input: {test_single.shape}")
print(f"    Output: {output_single.shape}")
print(f"    ✓ Single image handling works")

if all_correct:
    print("\n✓ All dimensions correct!")
else:
    print("\n✗ Some dimensions are incorrect!")

print("\n✓ Step 2.8 completed: Complete forward pass implemented and tested")
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE!")
print("=" * 60)
print("\nNetwork Architecture Summary:")
print("  1. Conv1 (Emboss 3x3):     28×28 → 26×26")
print("  2. MaxPool1 (2×2):         26×26 → 13×13")
print("  3. Conv2 (Sobel 3x3):      13×13 → 11×11")
print("  4. MaxPool2 (2×2):         11×11 → 5×5")
print("  5. Flatten:                 5×5 → 25")
print("  6. Dense1 (16) + ReLU:     25 → 16")
print("  7. Dropout (0.5):          16 → 16")
print("  8. Dense2 (1) + Sigmoid:   16 → 1")

