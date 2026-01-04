# PneumoniaMNIST Binary Classification - Actionable Roadmap

## Overview
Build a complete NumPy-based CNN pipeline for binary classification of chest X-ray images (28x28 grayscale) to detect pneumonia.

---

## Phase 1: Setup & Data Preparation

### Step 1.1: Environment Setup
- [ ] Create project structure
- [ ] Install required packages: `numpy`, `matplotlib`, `tqdm`, `scikit-learn`
- [ ] Download PneumoniaMNIST dataset (pneumoniamnist.npz)
  - Source: https://github.com/MedMNIST/MedMNIST
  - Or use: `pip install medmnist` and load programmatically

### Step 1.2: Data Loading & Initial Exploration
- [ ] Load `pneumoniamnist.npz` file
- [ ] Extract arrays: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels`
- [ ] Print dataset shapes and basic statistics
- [ ] Verify image dimensions (should be 28x28 grayscale)
- [ ] Check label encoding (0=Normal, 1=Pneumonia)

### Step 1.3: Data Normalization
- [ ] Normalize images: divide by 255.0 to get range [0, 1]
- [ ] Verify normalization (min=0, max=1)
- [ ] Create directory: `chatgpt data normalized/`
- [ ] Save normalized arrays to the directory

### Step 1.4: Data Visualization & EDA
- [ ] Visualize 5-10 sample images from each class (Normal vs Pneumonia)
- [ ] Plot pixel intensity histograms for each class
- [ ] Compute class distributions:
  - Count samples per class in train/val/test
  - Calculate percentages
  - Report findings
- [ ] Calculate class weights:
  - Formula: `weight_class_0 = total_samples / (2 * class_0_samples)`
  - Formula: `weight_class_1 = total_samples / (2 * class_1_samples)`
  - Store for use in weighted loss function

---

## Phase 2: Network Architecture Implementation

### Step 2.1: Define Kernel Functions
- [ ] Implement 3x3 Emboss kernel:
  ```
  [[-2, -1,  0],
   [-1,  1,  1],
   [ 0,  1,  2]]
  ```
- [ ] Implement 3x3 Sobel kernel (for edge detection):
  ```
  Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
  ```
  (Choose one or combine - document choice)

### Step 2.2: Convolution Layer Implementation
- [ ] Implement `conv2d()` function:
  - Input: image (H, W) or batch (N, H, W)
  - Kernel: (K, K)
  - Output: (H-K+1, W-K+1) or (N, H-K+1, W-K+1)
  - Handle padding if needed (or document no padding)
- [ ] Test on sample image to verify dimensions

### Step 2.3: Max Pooling Implementation
- [ ] Implement `max_pool2d()` function:
  - Input: (H, W) or (N, H, W)
  - Pool size: 2x2
  - Stride: 2
  - Output: (H/2, W/2) or (N, H/2, W/2)
- [ ] Track max indices for backpropagation

### Step 2.4: Activation Functions
- [ ] Implement `relu()` and `relu_derivative()`
- [ ] Implement `sigmoid()` and `sigmoid_derivative()`

### Step 2.5: Flatten Layer
- [ ] Implement flatten operation
- [ ] Track original shape for potential unflatten (if needed)

### Step 2.6: Dense Layer Implementation
- [ ] Implement forward pass: `W @ x + b`
- [ ] Implement backward pass (gradient computation)
- [ ] Initialize weights (Xavier/He initialization or small random)

### Step 2.7: Dropout Implementation
- [ ] Implement dropout forward (training vs inference mode)
- [ ] Implement dropout backward (mask gradients)

### Step 2.8: Complete Forward Pass
- [ ] Chain all layers:
  1. Conv1 (Emboss) → (28x28 → 26x26)
  2. MaxPool1 → (26x26 → 13x13)
  3. Conv2 (Sobel) → (13x13 → 11x11)
  4. MaxPool2 → (11x11 → 5x5)
  5. Flatten → (25,)
  6. Dense(16) + ReLU → (16,)
  7. Dropout(0.5) → (16,)
  8. Dense(1) + Sigmoid → (1,)
- [ ] Verify dimensions at each step
- [ ] Document dimension flow

---

## Phase 3: Loss Function & Optimizer

### Step 3.1: Weighted Binary Cross-Entropy Loss
- [ ] Implement W-BCE loss:
  ```
  loss = -[w0 * y * log(p) + w1 * (1-y) * log(1-p)]
  ```
  where `w0` and `w1` are class weights
- [ ] Implement loss derivative for backpropagation

### Step 3.2: SGD Optimizer
- [ ] Implement SGD update rule:
  ```
  θ = θ - learning_rate * gradient
  ```
- [ ] Set learning rate (start with 0.01, tune if needed)

---

## Phase 4: Backward Propagation

### Step 4.1: Backward Pass Implementation
- [ ] Implement backward pass for each layer:
  - Output layer (sigmoid + W-BCE)
  - Dropout backward
  - Dense layer backward
  - Flatten backward (if needed)
  - MaxPool backward (using stored indices)
  - Conv2 backward
  - Conv1 backward
- [ ] Verify gradient shapes match parameter shapes

### Step 4.2: Gradient Checking (Optional but Recommended)
- [ ] Implement numerical gradient check
- [ ] Verify backward pass correctness

---

## Phase 5: Training Loop

### Step 5.1: Training Infrastructure
- [ ] Create training function
- [ ] Implement batch iteration (mini-batch SGD)
- [ ] Set batch size (e.g., 32 or 64)
- [ ] Initialize all parameters (conv kernels, dense weights/biases)

### Step 5.2: Training Loop
- [ ] For each epoch (max 50):
  - Shuffle training data
  - For each batch:
    - Forward pass
    - Compute loss
    - Backward pass
    - Update parameters (SGD)
  - Evaluate on validation set
  - Track training/validation loss
  - Track validation metrics (accuracy, precision, recall, F1)

### Step 5.3: Early Stopping
- [ ] Implement early stopping:
  - Monitor validation loss
  - Patience = 5 epochs
  - Save best model (or track best epoch)
  - Stop if no improvement for 5 epochs

### Step 5.4: Metrics Tracking
- [ ] Implement metric functions:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Specificity
- [ ] Store metrics per epoch

---

## Phase 6: Visualization During Training

### Step 6.1: Loss Plots
- [ ] Plot training loss vs epoch
- [ ] Plot validation loss vs epoch
- [ ] Combine in one figure with legend

### Step 6.2: Metrics Plots
- [ ] Plot validation metrics over epochs:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- [ ] Create subplot or separate figures

---

## Phase 7: Evaluation

### Step 7.1: Final Model Evaluation
- [ ] Load best model (or use final model)
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Compute all metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC (Area Under ROC Curve)
  - Specificity

### Step 7.2: Confusion Matrix
- [ ] Generate confusion matrix
- [ ] Visualize with matplotlib
- [ ] Add labels and values

### Step 7.3: ROC Curve
- [ ] Compute ROC curve (TPR vs FPR)
- [ ] Calculate AUC
- [ ] Plot ROC curve with AUC value

---

## Phase 8: Reporting & Documentation

### Step 8.1: Code Documentation
- [ ] Add comments explaining each function
- [ ] Document dimension transformations
- [ ] Explain kernel choices (Emboss, Sobel)
- [ ] Document hyperparameters

### Step 8.2: Create Report (PDF)
- [ ] **Architecture Section:**
  - Diagram or description of network
  - Dimension flow table
  - Explanation of each layer's role
  - Kernel explanations (Emboss for texture, Sobel for edges)

- [ ] **EDA Section:**
  - Dataset statistics
  - Class distribution plots
  - Sample images
  - Pixel intensity histograms
  - Observations

- [ ] **Training Section:**
  - Training/validation loss plots
  - Validation metrics over epochs
  - Training hyperparameters
  - Early stopping behavior

- [ ] **Evaluation Section:**
  - Final metrics table
  - Confusion matrix
  - ROC curve
  - Discussion of which class was harder to classify

- [ ] **Discussion Section:**
  - Challenges encountered
  - Suggestions for improvement
  - Architecture choices rationale

---

## Phase 9: Final Deliverables Checklist

### Step 9.1: Code File
- [ ] Complete `.ipynb` or `.py` file with:
  - All code implementations
  - All visualizations
  - All outputs/results
  - Clear structure and comments

### Step 9.2: Report PDF
- [ ] All sections completed
- [ ] All plots included
- [ ] Professional formatting
- [ ] Clear explanations

### Step 9.3: Data Directory
- [ ] Verify `chatgpt data normalized/` exists
- [ ] Verify normalized arrays are saved

---

## Implementation Tips

### Dimension Tracking
- Keep a dimension log: document input/output shapes at each layer
- Example:
  - Input: (N, 28, 28)
  - Conv1: (N, 26, 26)
  - MaxPool1: (N, 13, 13)
  - Conv2: (N, 11, 11)
  - MaxPool2: (N, 5, 5)
  - Flatten: (N, 25)
  - Dense1: (N, 16)
  - Dense2: (N, 1)

### Testing Strategy
- Test each layer independently before chaining
- Use small test cases (e.g., 2x2 image) to verify logic
- Print shapes at each step during initial runs

### Common Pitfalls
- Matrix multiplication dimension mismatches
- Forgetting to apply activation functions
- Incorrect gradient shapes
- Not handling batch dimension correctly
- Dropout active during inference

### Performance Considerations
- Use vectorized operations (NumPy)
- Consider batch processing for efficiency
- Monitor memory usage with large batches

---

## Estimated Time Breakdown

- Phase 1 (Data Prep): 2-3 hours
- Phase 2 (Architecture): 4-5 hours
- Phase 3-4 (Loss & Backprop): 3-4 hours
- Phase 5 (Training): 2-3 hours
- Phase 6-7 (Visualization & Eval): 2-3 hours
- Phase 8 (Reporting): 2-3 hours

**Total: ~15-21 hours**

---

## Next Steps

1. Start with Phase 1: Set up environment and load data
2. Work through phases sequentially
3. Test each component before moving to the next
4. Keep dimension tracking document updated
5. Save intermediate results (normalized data, trained model)

Good luck! 🚀

