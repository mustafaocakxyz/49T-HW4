# Quick Start Guide

## Immediate Next Steps

### 1. Get the Dataset
```bash
# Option 1: Install medmnist package
pip install medmnist

# Option 2: Download directly from GitHub
# Visit: https://github.com/MedMNIST/MedMNIST
# Download pneumoniamnist.npz
```

### 2. Install Dependencies
```bash
pip install numpy matplotlib tqdm scikit-learn
```

### 3. Start Implementation
- Open `IMPLEMENTATION_TEMPLATE.py` or create a new Jupyter notebook
- Follow `ROADMAP.md` step by step
- Refer to `DIMENSION_TRACKING.md` for dimension checks

## Key Formulas to Remember

### Convolution Output Size
```
Output = (Input - Kernel + 2*Padding) / Stride + 1
For no padding, stride=1: Output = Input - Kernel + 1
```

### Pooling Output Size
```
Output = (Input - Pool) / Stride + 1
For 2x2 pool, stride=2: Output = Input / 2
```

### Weighted Binary Cross-Entropy
```
Loss = -[w0 * y * log(p) + w1 * (1-y) * log(1-p)]
where:
  w0 = weight for class 0 (Normal)
  w1 = weight for class 1 (Pneumonia)
  y = true label (0 or 1)
  p = predicted probability
```

### Class Weights
```
w0 = total_samples / (2 * num_class_0_samples)
w1 = total_samples / (2 * num_class_1_samples)
```

## Critical Dimension Checks

Your network should follow this dimension flow:
```
Input:      (N, 28, 28)
Conv1:      (N, 26, 26)  [28-3+1]
MaxPool1:   (N, 13, 13)  [26/2]
Conv2:      (N, 11, 11)  [13-3+1]
MaxPool2:   (N, 5, 5)    [11/2]
Flatten:    (N, 25)      [5*5]
Dense1:     (N, 16)
Dropout:    (N, 16)
Dense2:     (N, 1)
```

## Testing Strategy

1. **Test each layer independently** before chaining
2. **Use small test cases**: Start with 2x2 or 4x4 images
3. **Print shapes** at every step during initial development
4. **Verify gradients**: Check that gradient shapes match parameter shapes

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Dimension mismatch | Check matrix multiplication shapes |
| Gradient shape error | Ensure gradients match parameter shapes |
| Pooling rounding | Use integer division (//) |
| Dropout in inference | Set training=False during evaluation |
| Memory issues | Reduce batch size |

## Evaluation Metrics Checklist

Make sure you compute:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ AUC (Area Under ROC Curve)
- ✅ Specificity = TN / (TN + FP)

## Report Sections Checklist

Your PDF report should include:
1. ✅ Architecture explanation with dimensions
2. ✅ EDA (sample images, histograms, class distribution)
3. ✅ Training plots (loss curves, metrics over epochs)
4. ✅ Evaluation results (metrics table, confusion matrix, ROC curve)
5. ✅ Discussion (which class was harder, challenges, improvements)

## Time Management Tips

- **Don't skip testing**: Test each component before moving on
- **Save frequently**: Save normalized data, model checkpoints
- **Document as you go**: Write explanations while code is fresh
- **Start report early**: Don't wait until the end

## Getting Help

If stuck:
1. Check dimension tracking document
2. Verify formulas are correct
3. Test with smaller examples
4. Print intermediate values to debug

Good luck! 🎯

