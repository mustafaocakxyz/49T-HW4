import numpy as np

np.random.seed(42)

def weighted_binary_cross_entropy(y_true, y_pred, class_weights):
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

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, parameters, gradients):
        updated_params = []
        for param, grad in zip(parameters, gradients):
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)
        return updated_params

