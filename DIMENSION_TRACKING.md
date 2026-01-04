# Dimension Tracking Reference

## Input Specifications
- Image size: 28x28 grayscale
- Batch size: N (e.g., 32, 64)
- Input shape: (N, 28, 28) or (N, 1, 28, 28) if using channel dimension

## Layer-by-Layer Dimension Flow

### Forward Pass Dimensions

| Layer | Input Shape | Operation | Output Shape | Notes |
|-------|-------------|-----------|--------------|-------|
| Input | (N, 28, 28) | - | (N, 28, 28) | Batch of images |
| Conv1 (Emboss 3x3) | (N, 28, 28) | Conv, no padding | (N, 26, 26) | H-K+1 = 28-3+1 = 26 |
| MaxPool1 (2x2) | (N, 26, 26) | Pool, stride=2 | (N, 13, 13) | 26/2 = 13 |
| Conv2 (Sobel 3x3) | (N, 13, 13) | Conv, no padding | (N, 11, 11) | 13-3+1 = 11 |
| MaxPool2 (2x2) | (N, 11, 11) | Pool, stride=2 | (N, 5, 5) | 11/2 = 5 (rounded down) |
| Flatten | (N, 5, 5) | Reshape | (N, 25) | 5*5 = 25 |
| Dense1 (16 units) | (N, 25) | Linear + ReLU | (N, 16) | W: (25, 16), b: (16,) |
| Dropout (0.5) | (N, 16) | Mask | (N, 16) | Same shape |
| Dense2 (1 unit) | (N, 16) | Linear + Sigmoid | (N, 1) | W: (16, 1), b: (1,) |

## Backward Pass Dimension Checks

### Gradient Shapes Must Match Parameter Shapes

| Layer | Parameter | Shape | Gradient Shape |
|-------|-----------|-------|----------------|
| Dense2 | W | (16, 1) | (16, 1) |
| Dense2 | b | (1,) | (1,) |
| Dense1 | W | (25, 16) | (25, 16) |
| Dense1 | b | (16,) | (16,) |
| Conv2 | Kernel | (3, 3) | (3, 3) |
| Conv1 | Kernel | (3, 3) | (3, 3) |

## Convolution Dimension Formula

**Output size = (Input_size - Kernel_size + 2*Padding) / Stride + 1**

For our case (no padding, stride=1):
- Output = Input_size - Kernel_size + 1

## Pooling Dimension Formula

**Output size = (Input_size - Pool_size) / Stride + 1**

For 2x2 pooling with stride=2:
- Output = Input_size / 2 (integer division)

## Matrix Multiplication Dimensions

- Dense forward: `(N, in_features) @ (in_features, out_features) = (N, out_features)`
- Dense backward: 
  - dW: `(in_features, N) @ (N, out_features) = (in_features, out_features)`
  - db: sum over batch dimension
  - dx: `(N, out_features) @ (out_features, in_features) = (N, in_features)`

## Common Dimension Errors to Avoid

1. **Forgetting batch dimension**: Always account for N in first dimension
2. **Transpose errors**: Matrix multiplication requires correct transpose
3. **Pooling rounding**: 11/2 = 5 (not 5.5, use floor division)
4. **Flatten shape**: Ensure correct reshape (5, 5) → (25,)
5. **Gradient accumulation**: Sum gradients over batch dimension

## Testing Dimensions

Use this test case to verify:
```python
# Test with batch_size=2
test_input = np.random.randn(2, 28, 28)
# After each layer, print shape and verify
```

