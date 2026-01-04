import numpy as np

np.random.seed(42)

def conv2d_backward(grad_output, input_data, kernel):
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
    return grad_output.reshape(original_shape)

