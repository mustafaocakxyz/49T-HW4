import numpy as np

np.random.seed(42)

def get_emboss_kernel():
    emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    return emboss

def get_sobel_kernel():
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return sobel_x

def conv2d(input_data, kernel, stride=1, padding=0):
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
    original_shape = input_data.shape
    if input_data.ndim == 2:
        flattened = input_data.flatten()
    else:
        N = input_data.shape[0]
        flattened = input_data.reshape(N, -1)
    return flattened, original_shape

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    s = sigmoid(x) if np.any(x < 0) or np.any(x > 1) else x
    return s * (1 - s)

class DenseLayer:
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

