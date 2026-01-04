# PneumoniaMNIST Binary Classification Report

**Author:** [Your Name]  
**Date:** [Date]  
**Course:** CMPE49T HW4

---

**Note on Figures**: All figures referenced in this report are located in the `phase1_plots/` directory. When converting this report to PDF, ensure all figures are properly embedded.

---

## 1. Introduction

### 1.1 Problem Statement
This project implements a complete pipeline for binary classification of chest X-ray images to detect pneumonia using a custom-built neural network implemented in NumPy. The goal is to classify 28x28 grayscale chest X-ray images into two classes: Normal (0) and Pneumonia (1).

### 1.2 Dataset Description
- **Dataset**: PneumoniaMNIST (part of MedMNIST collection)
- **Image Size**: 28x28 grayscale
- **Classes**: 
  - 0: Normal
  - 1: Pneumonia
- **Splits**: Pre-split into training, validation, and test subsets
- **Source**: https://github.com/MedMNIST/MedMNIST

---

## 2. Data Exploration and Analysis

### 2.1 Dataset Statistics

The PneumoniaMNIST dataset was successfully loaded and analyzed. The dataset contains:

- **Training Set**: 4,708 samples
- **Validation Set**: 524 samples
- **Test Set**: 624 samples
- **Total Samples**: 5,856 images
- **Image Dimensions**: 28×28 grayscale
- **Pixel Range**: 
  - Before normalization: 0-255 (integer values)
  - After normalization: 0.0-1.0 (float32 values)

**Pre-normalization Statistics:**
- Training images: Min=0, Max=255, Mean=145.84
- Validation images: Min=0, Max=255, Mean=145.14
- Test images: Min=0, Max=254, Mean=143.81

The dataset is pre-split, ensuring no data leakage between training, validation, and test sets.

### 2.2 Class Distribution

The class distribution across different splits is shown in Table 1 and Figure 1.

**Table 1: Class Distribution Across Splits**

| Split | Normal (0) | Pneumonia (1) | Total | Normal % | Pneumonia % |
|-------|------------|---------------|-------|----------|--------------|
| Training | 1,214 | 3,494 | 4,708 | 25.79% | 74.21% |
| Validation | 135 | 389 | 524 | 25.76% | 74.24% |
| Test | 234 | 390 | 624 | 37.50% | 62.50% |

**Figure 1: Class Distribution Visualization**
*[See: `phase1_plots/class_distribution.png`]*

**Key Observations:**

1. **Consistent Train/Val Distribution**: The training and validation sets maintain nearly identical class distributions (~25.8% Normal, ~74.2% Pneumonia), which is ideal for model development and validation.

2. **Test Set Distribution Mismatch**: The test set exhibits a different class distribution (37.5% Normal, 62.5% Pneumonia) compared to the training/validation sets. This is a common characteristic in medical datasets where:
   - Test sets may be curated from different sources or time periods
   - Test sets may be designed to be more balanced for evaluation
   - This difference should be considered when interpreting test set results

3. **Implications**: 
   - The model is trained on data with 74% Pneumonia cases
   - Test evaluation occurs on a more balanced distribution (62.5% Pneumonia)
   - This may affect the direct comparability of metrics, requiring careful interpretation

### 2.3 Class Imbalance Analysis

**Imbalance Ratio**: 2.878:1 (Pneumonia:Normal) in the training set

This significant class imbalance presents a challenge for model training:

- **Risk**: The model may develop a bias toward predicting the majority class (Pneumonia)
- **Impact**: Without mitigation, the model might achieve high accuracy by simply predicting Pneumonia for most cases, while performing poorly on the minority class (Normal)

**Mitigation Strategy**: Class weights are calculated and will be used in a Weighted Binary Cross-Entropy loss function:

- **Weight for Normal (0)**: 1.9390
- **Weight for Pneumonia (1)**: 0.6737

**Calculation Formula**:
```
weight_class = total_samples / (num_classes × class_samples)
```

The higher weight for the Normal class (1.9390) compensates for its underrepresentation, ensuring that misclassifying a Normal case contributes more to the loss than misclassifying a Pneumonia case.

### 2.4 Sample Images Visualization

**Figure 2: Sample Images from Each Class**
*[See: `phase1_plots/sample_images.png`]*

The visualization shows 5 sample images from each class (Normal on top row, Pneumonia on bottom row).

**Visual Observations:**

1. **Normal Chest X-rays**:
   - Generally show clearer lung fields
   - More uniform intensity distribution
   - Less visible opacities or infiltrates

2. **Pneumonia Cases**:
   - May show areas of increased opacity
   - Potential infiltrates or consolidations
   - Some cases show subtle differences from normal

3. **Resolution Limitation**:
   - The 28×28 resolution significantly limits detail visibility
   - Fine-grained features that radiologists use (e.g., subtle infiltrates, vessel patterns) are not clearly visible
   - This low resolution makes the classification task more challenging

4. **Inter-class Similarity**:
   - Some images from different classes appear visually similar at this resolution
   - This suggests the model will need to learn subtle patterns rather than obvious visual differences

### 2.5 Pixel Intensity Analysis

**Figure 3: Pixel Intensity Histograms**
*[See: `phase1_plots/pixel_histograms.png`]*

The pixel intensity histograms reveal the following statistics:

**Normal Class:**
- Mean intensity: 139.90
- Standard deviation: 46.82
- Distribution: Relatively normal distribution centered around 140

**Pneumonia Class:**
- Mean intensity: 147.90
- Standard deviation: 41.29
- Distribution: Slightly shifted toward higher intensities, with slightly tighter distribution

**Key Findings:**

1. **Overlapping Distributions**: The histograms show significant overlap between the two classes, indicating that:
   - Simple thresholding on pixel intensity would not be effective
   - The classification task requires learning spatial patterns, not just intensity values
   - Both classes span similar intensity ranges

2. **Subtle Differences**: 
   - Pneumonia cases have a slightly higher mean intensity (147.90 vs 139.90)
   - This difference is small relative to the standard deviations
   - The difference may reflect areas of consolidation or opacity in pneumonia cases

3. **Implications for Model**:
   - The model must learn spatial relationships and patterns
   - Convolutional operations will be crucial for capturing local features
   - Edge detection and texture analysis (via Emboss and Sobel kernels) should help distinguish classes

### 2.6 Data Preprocessing

**Normalization Procedure:**

All images were normalized to the range [0, 1] by dividing pixel values by 255.0 and converting to float32 format:

```python
normalized_images = images.astype(np.float32) / 255.0
```

**Normalization Verification:**

| Split | Min | Max | Mean |
|-------|-----|-----|------|
| Training | 0.0000 | 1.0000 | 0.5719 |
| Validation | 0.0000 | 1.0000 | 0.5692 |
| Test | 0.0000 | 0.9961 | 0.5640 |

✓ All values confirmed to be in [0, 1] range  
✓ Normalization successful across all splits

**Data Storage:**

- **Normalized Data**: Saved to `chatgpt data normalized/normalized_data.npz`
  - Contains: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels`
- **Class Weights**: Saved to `chatgpt data normalized/class_weights.npz`
  - Contains: `weight_class_0`, `weight_class_1`

**Rationale for Normalization:**

1. **Numerical Stability**: Prevents large pixel values from dominating gradient calculations
2. **Activation Function Compatibility**: Sigmoid activation works best with inputs in [0, 1] range
3. **Training Efficiency**: Normalized inputs allow for more stable learning rates
4. **Standard Practice**: Common preprocessing step in deep learning pipelines

### 2.7 Summary of EDA Findings

**Dataset Characteristics:**
- ✓ Dataset successfully loaded and verified
- ✓ Significant class imbalance (2.88:1 ratio) identified and addressed
- ✓ Test set distribution differs from train/val (expected in medical datasets)
- ✓ Images properly normalized and stored

**Key Challenges Identified:**
1. **Class Imbalance**: Mitigated with class weights
2. **Low Resolution**: 28×28 limits detail visibility
3. **Similar Pixel Intensities**: Requires spatial pattern learning
4. **Test Distribution Mismatch**: Requires careful metric interpretation

**Prepared for Training:**
- ✓ Normalized data ready for network input
- ✓ Class weights calculated for weighted loss
- ✓ All visualizations generated for report
- ✓ Data quality verified (no missing values, correct ranges)

---

## 3. Network Architecture

### 3.1 Architecture Overview

The network is a custom-built Convolutional Neural Network (CNN) implemented entirely in NumPy, without using high-level deep learning frameworks. The architecture is designed to extract spatial features from 28×28 grayscale chest X-ray images and classify them as Normal or Pneumonia.

**Network Structure:**

The network consists of 8 layers organized into feature extraction and classification stages:

**Feature Extraction Stage:**
1. **Convolution Layer 1**: 3×3 Emboss kernel (texture enhancement)
2. **Max Pooling Layer 1**: 2×2 pooling with stride 2 (dimensionality reduction)
3. **Convolution Layer 2**: 3×3 Sobel kernel (edge detection)
4. **Max Pooling Layer 2**: 2×2 pooling with stride 2 (further dimensionality reduction)

**Classification Stage:**
5. **Flatten Layer**: Reshapes 2D feature maps to 1D vector
6. **Dense Layer 1**: 16 units with ReLU activation (feature learning)
7. **Dropout Layer**: Rate = 0.5 (regularization)
8. **Dense Layer 2**: 1 unit with sigmoid activation (binary classification output)

**Design Rationale:**

- **Two-stage convolution**: First layer (Emboss) enhances texture patterns, second layer (Sobel) detects edges and boundaries
- **Progressive dimensionality reduction**: Two pooling layers reduce spatial dimensions from 28×28 to 5×5, reducing parameters while preserving important features
- **Small dense layer**: 16 units chosen to balance model capacity and overfitting risk
- **Dropout regularization**: 0.5 rate helps prevent overfitting in the dense layers
- **Sigmoid output**: Produces probability scores in [0, 1] range for binary classification

### 3.2 Dimension Flow

The dimension flow through the network is carefully tracked to ensure correct matrix operations. The following table shows the transformation at each layer:

**Table 2: Network Dimension Flow**

| Layer | Input Shape | Output Shape | Transformation | Formula |
|-------|-------------|--------------|---------------|---------|
| Input | (N, 28, 28) | (N, 28, 28) | - | - |
| Conv1 (Emboss 3×3) | (N, 28, 28) | (N, 26, 26) | Convolution, no padding | H_out = H - K + 1 |
| MaxPool1 (2×2) | (N, 26, 26) | (N, 13, 13) | Max pooling, stride=2 | H_out = H / 2 |
| Conv2 (Sobel 3×3) | (N, 13, 13) | (N, 11, 11) | Convolution, no padding | H_out = H - K + 1 |
| MaxPool2 (2×2) | (N, 11, 11) | (N, 5, 5) | Max pooling, stride=2 | H_out = H / 2 |
| Flatten | (N, 5, 5) | (N, 25) | Reshape | 5 × 5 = 25 |
| Dense1 (16 units) | (N, 25) | (N, 16) | Linear + ReLU | W: (25, 16), b: (16,) |
| Dropout (0.5) | (N, 16) | (N, 16) | Mask (training) | Same shape |
| Dense2 (1 unit) | (N, 16) | (N, 1) | Linear + Sigmoid | W: (16, 1), b: (1,) |

**Dimension Verification:**

All dimensions were verified through testing:
- ✓ Convolution correctly reduces dimensions: 28 → 26 → 13 → 11 → 5
- ✓ Pooling correctly halves dimensions: 26 → 13, 11 → 5
- ✓ Flatten correctly converts 5×5 to 25 features
- ✓ Dense layers correctly transform: 25 → 16 → 1
- ✓ Final output shape: (N, 1) for batch of N images

**Total Parameters:**

- Conv1: 3×3 = 9 parameters (fixed kernel)
- Conv2: 3×3 = 9 parameters (fixed kernel)
- Dense1: 25×16 + 16 = 416 parameters
- Dense2: 16×1 + 1 = 17 parameters
- **Total trainable parameters**: 433 (excluding fixed convolution kernels)

### 3.3 Kernel Choices

The network uses two fixed convolution kernels, each serving a specific purpose in feature extraction:

**Emboss Kernel (Layer 1):**

```
[[-2, -1,  0],
 [-1,  1,  1],
 [ 0,  1,  2]]
```

**Purpose**: Texture enhancement and depth perception

**Rationale**:
- The emboss filter creates a 3D embossed effect by highlighting intensity transitions
- In X-ray images, this helps emphasize subtle variations in tissue density
- Pneumonia often presents as areas of increased opacity (consolidation), which the emboss filter can help highlight
- The asymmetric pattern enhances directional texture information

**Effect**: Transforms smooth intensity gradients into more pronounced texture patterns, making density variations more detectable.

**Sobel Kernel (Layer 2):**

```
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
```

**Purpose**: Edge detection and boundary identification

**Rationale**:
- The Sobel X kernel detects horizontal edges (vertical intensity gradients)
- In chest X-rays, edges correspond to boundaries between different tissue types, lung borders, and pathological structures
- Pneumonia often manifests as irregular opacities with distinct boundaries
- Edge detection helps identify structural abnormalities and consolidation patterns

**Effect**: Highlights regions with strong intensity gradients, effectively identifying boundaries and structural features that may indicate pathological changes.

**Kernel Selection Strategy**:

1. **Fixed vs. Learnable**: Using fixed kernels (Emboss, Sobel) instead of learnable filters was chosen to:
   - Simplify implementation (no kernel gradients needed)
   - Provide interpretable feature extraction
   - Reduce computational complexity
   - Ensure stable, predictable feature extraction

2. **Order**: Emboss before Sobel allows:
   - First enhancing texture patterns
   - Then detecting edges in the enhanced features
   - Progressive feature refinement

3. **Alternative Considerations**: 
   - Learnable kernels could potentially adapt better to the specific dataset
   - However, fixed kernels provide good baseline features and are computationally efficient
   - The dense layers can learn to combine these fixed features effectively

### 3.4 Activation Functions

The network uses two activation functions, each chosen for specific roles:

**ReLU (Rectified Linear Unit) - Dense Layer 1**

**Function**: `f(x) = max(0, x)`

**Derivative**: `f'(x) = 1 if x > 0, else 0`

**Purpose**: 
- Introduces non-linearity into the network
- Addresses vanishing gradient problem (gradient is 1 for positive inputs)
- Enables sparse activations (many neurons output 0)
- Computationally efficient

**Why ReLU for Hidden Layer**:
- Prevents vanishing gradients that can occur with sigmoid/tanh in deep networks
- Sparse activations help with feature selection
- He initialization (used for Dense1) is designed for ReLU activations
- Empirically shown to work well in practice

**Sigmoid - Output Layer**

**Function**: `f(x) = 1 / (1 + exp(-x))`

**Derivative**: `f'(x) = f(x) × (1 - f(x))`

**Purpose**:
- Produces output in [0, 1] range, interpretable as probability
- Smooth, differentiable function
- Appropriate for binary classification

**Why Sigmoid for Output**:
- Binary classification requires probability output
- Sigmoid naturally maps any real number to [0, 1]
- Smooth gradient flow (unlike step function)
- Standard choice for binary classification tasks

**Implementation Details**:

- **Numerical Stability**: Sigmoid implementation clips input to [-500, 500] to prevent overflow
- **Verification**: Output verified to be in [0, 1] range for all test cases
- **Gradient Flow**: Both activation functions have well-defined derivatives for backpropagation

### 3.5 Layer Implementation Details

**Convolution Layer**:
- Supports both single images and batches
- No padding used (padding=0) to reduce dimensions
- Stride=1 for dense feature extraction
- Handles dimension reduction: (H, W) → (H-K+1, W-K+1) for K×K kernel

**Max Pooling Layer**:
- 2×2 pooling window with stride=2
- Reduces dimensions by factor of 2
- Tracks indices of maximum values for backpropagation
- Helps with translation invariance and dimensionality reduction

**Flatten Layer**:
- Converts 2D feature maps to 1D vectors
- Stores original shape for potential unflattening
- Essential bridge between convolutional and dense layers

**Dense Layer**:
- Fully connected layer with learnable weights and biases
- **Initialization**:
  - Dense1: He initialization (optimal for ReLU)
  - Dense2: Xavier/Glorot initialization (optimal for sigmoid)
- Supports forward and backward passes
- Gradient computation verified for correct shapes

**Dropout Layer**:
- Rate = 0.5 (50% of neurons randomly set to zero during training)
- Scales output by 1/(1-rate) = 2.0 during training to maintain expected value
- No dropout during inference (training=False)
- Helps prevent overfitting by reducing co-adaptation of neurons

### 3.6 Architecture Verification

**Testing Results**:
- ✓ All layers produce correct output dimensions
- ✓ Forward pass works for both single images and batches
- ✓ Activation functions produce valid outputs (sigmoid in [0, 1])
- ✓ Dropout behaves correctly in training vs. inference modes
- ✓ Dimension flow verified at each layer
- ✓ No dimension mismatches or errors

**Ready for Training**:
- ✓ All components implemented and tested
- ✓ Forward pass complete and verified
- ✓ Backward propagation implemented and verified
- ✓ Network can process normalized input images correctly
- ✓ All gradients computed correctly with proper shapes

---

## 4. Training Procedure

### 4.1 Loss Function

**Weighted Binary Cross-Entropy Loss**:

The loss function addresses the significant class imbalance (2.88:1 ratio) in the training dataset by assigning higher weight to the minority class (Normal).

**Mathematical Formulation**:
```
Loss = -[w₀ × y × log(p) + w₁ × (1-y) × log(1-p)]
```

where:
- **w₀** = 1.9390 (weight for Normal class)
- **w₁** = 0.6737 (weight for Pneumonia class)
- **y** = true label (0 or 1)
- **p** = predicted probability from sigmoid output

**Gradient with Respect to Prediction**:
```
dL/dp = -[w₀ × y / p - w₁ × (1-y) / (1-p)]
```

**Implementation Details**:

1. **Numerical Stability**: Predictions are clipped to [ε, 1-ε] where ε = 1e-15 to prevent log(0) errors
2. **Class Weight Calculation**: Weights computed as `weight = total_samples / (2 × class_samples)`
3. **Loss Averaging**: Loss is averaged over the batch for stable gradients
4. **Gradient Computation**: Gradient is computed analytically and averaged over batch

**Verification Results**:
- ✓ Loss correctly increases with worse predictions (tested: perfect predictions → 0.0198, poor predictions → 3.0081)
- ✓ Gradient direction verified: negative gradient when prediction is too low, less negative when closer to target
- ✓ Loss function properly penalizes misclassification of minority class more heavily

**Rationale**:
- The higher weight for Normal class (1.9390 vs 0.6737) ensures that misclassifying a Normal case contributes more to the loss
- This prevents the model from simply predicting the majority class (Pneumonia) for all cases
- The weighted loss naturally balances the learning process without requiring class balancing techniques

### 4.2 Optimizer

**Stochastic Gradient Descent (SGD)**:

**Update Rule**:
```
θ = θ - learning_rate × gradient
```

where:
- **θ** = model parameters (weights and biases)
- **learning_rate** = step size for parameter updates
- **gradient** = computed gradient from backpropagation

**Implementation Details**:

- **Algorithm**: Standard SGD without momentum or adaptive learning rates
- **Learning Rate**: 0.01 (default, can be tuned during training)
- **Update Process**: Parameters updated after each batch using computed gradients
- **Batch Processing**: Gradients averaged over batch before parameter update

**Verification**:
- ✓ Update rule verified: parameters correctly updated as `θ_new = θ_old - lr × grad`
- ✓ Works correctly with different learning rates (tested: 0.01, 0.1, 1.0)
- ✓ Handles multiple parameter arrays (weights and biases) correctly

**Learning Rate Selection**:
- Initial learning rate of 0.01 chosen as a conservative starting point
- Can be adjusted based on training dynamics (loss convergence, validation performance)
- Lower learning rates provide more stable but slower convergence
- Higher learning rates may cause instability or overshooting

### 4.3 Backward Propagation

**Implementation Overview**:

Backward propagation computes gradients for all trainable parameters by applying the chain rule through the network layers.

**Gradient Flow Through Layers**:

1. **Output Layer (Sigmoid + Loss)**:
   - Gradient from loss function: `dL/dp` (from weighted BCE)
   - Gradient through sigmoid: `dL/dx = dL/dp × p × (1-p)`
   - Propagated to Dense2 layer

2. **Dense Layer 2**:
   - Gradient w.r.t. weights: `dL/dW = x^T × grad_output`
   - Gradient w.r.t. bias: `dL/db = sum(grad_output, axis=0)`
   - Gradient w.r.t. input: `dL/dx = grad_output × W^T`
   - Propagated to Dropout layer

3. **Dropout Layer**:
   - Gradient masked by dropout mask: `grad_input = grad_output × mask / (1-rate)`
   - Only propagates gradient to neurons that were active during forward pass
   - Maintains expected gradient magnitude through scaling

4. **ReLU Activation**:
   - Gradient: `dL/dx = grad_output × (x > 0)`
   - Zero gradient for negative inputs (dead neurons)
   - Propagated to Dense1 layer

5. **Dense Layer 1**:
   - Same gradient computation as Dense2
   - Propagated to Flatten layer

6. **Flatten Layer**:
   - Gradient reshaped to original dimensions: `grad_input = grad_output.reshape(original_shape)`
   - Propagated to MaxPool2 layer

7. **Max Pooling Layer 2**:
   - Gradient only propagated to positions that were maximum during forward pass
   - Uses stored indices from forward pass: `grad_input[indices] = grad_output`
   - All other positions receive zero gradient
   - Propagated to Conv2 layer

8. **Convolution Layer 2 (Sobel)**:
   - Gradient w.r.t. input: full convolution with flipped kernel
   - Gradient w.r.t. kernel: sum over input regions weighted by gradient
   - Note: Kernel gradients computed but not used (kernels are fixed)
   - Propagated to MaxPool1 layer

9. **Max Pooling Layer 1**:
   - Same as MaxPool2
   - Propagated to Conv1 layer

10. **Convolution Layer 1 (Emboss)**:
    - Same as Conv2
    - Note: Kernel gradients computed but not used (kernels are fixed)

**Gradient Verification**:

- ✓ All gradient shapes verified to match parameter shapes
- ✓ Gradient flow tested through complete network
- ✓ Numerical gradient checking implemented (some numerical precision differences expected)
- ✓ Backward pass correctly handles batch processing

**Key Implementation Details**:

- **Fixed Kernels**: Emboss and Sobel kernels are fixed (not trainable), so their gradients are computed but not used for updates
- **Max Pooling Indices**: Indices from forward pass are stored and used to route gradients correctly
- **Batch Averaging**: Gradients are averaged over the batch dimension before parameter updates
- **Numerical Stability**: Careful handling of division operations and clipping to prevent numerical issues

### 4.4 Training Configuration

**Hyperparameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 32 | Balances memory usage and gradient stability |
| Learning Rate | 0.01 | Conservative starting point for stable convergence |
| Max Epochs | 50 | Sufficient for convergence with early stopping |
| Early Stopping Patience | 5 | Prevents overfitting while allowing recovery |
| Optimizer | SGD | Simple, effective for this architecture |
| Loss Function | Weighted Binary Cross-Entropy | Addresses class imbalance |

**Training Procedure**:

1. **Data Preparation**:
   - Load normalized images and labels from `chatgpt data normalized/normalized_data.npz`
   - Load class weights: Normal=1.9390, Pneumonia=0.6737
   - Split data: 4,708 training, 524 validation, 624 test samples

2. **Batch Creation**:
   - Shuffle training data at the start of each epoch
   - Create batches of size 32
   - Last batch may be smaller if dataset size is not divisible by batch size

3. **Training Loop (per epoch)**:
   - For each batch:
     - **Forward Pass**: Compute predictions through all layers
     - **Loss Computation**: Calculate weighted binary cross-entropy loss
     - **Backward Pass**: Compute gradients through all layers
     - **Parameter Update**: Update dense layer weights and biases using SGD
   - Evaluate on validation set
   - Compute validation loss and metrics
   - Check early stopping condition

4. **Early Stopping**:
   - Monitor validation loss after each epoch
   - Track best validation loss and corresponding epoch
   - If validation loss doesn't improve for 5 consecutive epochs, stop training
   - Return to best model (epoch with lowest validation loss)

5. **Metrics Tracking**:
   - Compute metrics after each epoch on validation set:
     - Accuracy, Precision, Recall, F1 Score, Specificity, AUC
   - Store metrics for plotting and analysis

**Implementation Details**:

- **Backward Propagation**: Complete gradient flow through all layers:
  - Loss → Sigmoid → Dense2 → Dropout → ReLU → Dense1 → Flatten → MaxPool2 → Conv2 → MaxPool1 → Conv1
- **Parameter Updates**: Only dense layer parameters (weights and biases) are updated
  - Convolution kernels (Emboss, Sobel) are fixed and not updated
- **Dropout**: Applied only during training (training=True), disabled during validation (training=False)
- **Gradient Averaging**: Gradients averaged over batch before parameter updates

**Training Infrastructure**:

- ✓ Batch creation with shuffling
- ✓ Complete forward/backward pass implementation
- ✓ Parameter update mechanism
- ✓ Early stopping with patience mechanism
- ✓ Metrics computation (Accuracy, Precision, Recall, F1, Specificity, AUC)
- ✓ Validation evaluation after each epoch

### 4.5 Training Curves
**Figure 4: Training and Validation Loss Over Epochs**
*[See: `training_results/training_loss.png`]*

The training was completed for 50 epochs. The loss curves show:
- **Training Loss**: Decreased from 0.8036 (epoch 1) to 0.2825 (epoch 50)
- **Validation Loss**: Decreased from 0.4825 (epoch 1) to 0.2453 (epoch 50)
- **Best Epoch**: Epoch 48 with validation loss of 0.2376
- **Convergence**: Model converged steadily with validation loss decreasing throughout training
- **No Overfitting**: Validation loss continued to decrease alongside training loss, indicating good generalization

**Observations**:
- Steady decrease in both training and validation loss
- No significant gap between training and validation loss (good generalization)
- Best model found at epoch 48, just before the end of training
- Model did not trigger early stopping (patience=5), suggesting continued improvement

### 4.6 Validation Metrics Over Epochs
**Figure 5: Validation Metrics Over Epochs**
*[See: `training_results/validation_metrics.png`]*

The validation metrics show consistent improvement over training:

- **Accuracy**: Improved from 0.7424 (epoch 1) to 0.8779 (final)
- **Precision**: Improved from 0.8521 (epoch 1) to 0.8719 (final)
- **Recall**: Improved from 0.8521 (epoch 1) to 0.9794 (final)
- **F1 Score**: Improved from 0.8521 (epoch 1) to 0.9225 (final)

**Trend Analysis**:
- All metrics show steady upward trend
- Best performance achieved around epoch 48
- Metrics stabilized in later epochs with minor fluctuations
- High recall (0.9794) indicates model successfully identifies most Pneumonia cases

**Metrics Computation Implementation**:

All metrics are computed from the confusion matrix components:

- **True Positives (TP)**: Correctly predicted Pneumonia cases
- **False Positives (FP)**: Normal cases incorrectly predicted as Pneumonia
- **True Negatives (TN)**: Correctly predicted Normal cases
- **False Negatives (FN)**: Pneumonia cases incorrectly predicted as Normal

**Metric Formulas**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Overall correctness
- **Precision**: TP / (TP + FP) - Proportion of positive predictions that are correct
- **Recall (Sensitivity)**: TP / (TP + FN) - Proportion of actual positives correctly identified
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean of precision and recall
- **Specificity**: TN / (TN + FP) - Proportion of actual negatives correctly identified
- **AUC**: Area Under ROC Curve - Measures ability to distinguish between classes (computed using probability rankings)

**Early Stopping Implementation**:

The early stopping mechanism:
- **Initialization**: Sets best_loss = ∞, counter = 0, best_epoch = 0
- **Each Epoch**: 
  - Compares current validation loss with best_loss
  - If improvement (loss decreased by at least min_delta): resets counter, updates best_loss and best_epoch
  - If no improvement: increments counter
- **Stopping Condition**: Training stops when counter ≥ patience (5 epochs)
- **Best Model**: Returns to the epoch with lowest validation loss

**Benefits of Early Stopping**:
- Prevents overfitting by stopping when validation performance plateaus
- Saves computational resources by avoiding unnecessary epochs
- Automatically selects the best model without manual intervention
- Reduces risk of training on noise after convergence

---

## 5. Evaluation

### 5.1 Final Metrics

**Table 3: Final Evaluation Metrics**

| Metric | Validation Set | Test Set |
|--------|----------------|----------|
| Accuracy | 0.8779 | 0.7660 |
| Precision | 0.8719 | 0.7293 |
| Recall | 0.9794 | 0.9949 |
| F1 Score | 0.9225 | 0.8416 |
| AUC | 0.9642 | 0.9051 |
| Specificity | 0.5852 | 0.3846 |

**Key Findings**:

1. **High Recall**: The model achieves very high recall on both sets (0.9794 validation, 0.9949 test), meaning it correctly identifies almost all Pneumonia cases. This is crucial for medical diagnosis where missing a Pneumonia case (false negative) is more serious than a false positive.

2. **Moderate Specificity**: Lower specificity (0.5852 validation, 0.3846 test) indicates the model has more difficulty correctly identifying Normal cases, leading to more false positives.

3. **Good Overall Performance**: 
   - Validation accuracy: 87.79% - Strong performance on validation set
   - Test accuracy: 76.60% - Lower than validation, but still reasonable
   - AUC scores above 0.90 on both sets indicate good class separation

4. **Performance Gap**: Test set performance (76.60%) is lower than validation (87.79%), which may be due to:
   - Different class distribution in test set (37.5% Normal vs 25.8% in train/val)
   - Test set may contain more challenging cases
   - Model may have slightly overfit to validation set characteristics

### 5.2 Confusion Matrix

**Figure 6: Confusion Matrix - Test Set**
*[See: `training_results/confusion_matrix_test.png`]*

**Test Set Confusion Matrix**:

|                | Predicted Normal | Predicted Pneumonia |
|----------------|------------------|---------------------|
| **Actual Normal** | 90 (TN) | 144 (FP) |
| **Actual Pneumonia** | 2 (FN) | 388 (TP) |

**Analysis**:
- **True Positives (TP)**: 388 - Correctly identified Pneumonia cases
- **False Positives (FP)**: 144 - Normal cases incorrectly predicted as Pneumonia
- **True Negatives (TN)**: 90 - Correctly identified Normal cases
- **False Negatives (FN)**: 2 - Pneumonia cases missed (very low, which is good)

**Key Observations**:
- Model has very low false negatives (only 2), which is excellent for medical diagnosis
- Higher false positives (144) - model tends to predict Pneumonia when uncertain
- This aligns with the high recall (0.9949) and lower specificity (0.3846)

### 5.3 ROC Curve

**Figure 7: ROC Curve - Test Set**
*[See: `training_results/roc_curve_test.png`]*

**ROC Curve Analysis**:
- **AUC Score**: 0.9051 - Indicates good ability to distinguish between classes
- The curve shows strong performance with high true positive rate at low false positive rates
- The model performs significantly better than random (diagonal line)

**Interpretation**:
- AUC of 0.9051 means the model can correctly distinguish between Normal and Pneumonia cases 90.51% of the time
- The curve's shape indicates good sensitivity across different thresholds
- High AUC confirms the model's effectiveness despite the class imbalance

### 5.4 Per-Class Analysis

**Which class was harder to classify?**

**Answer: The Normal class was significantly harder to classify than the Pneumonia class.**

**Evidence**:

1. **Specificity (Normal class performance)**:
   - Validation: 0.5852 (58.52% of Normal cases correctly identified)
   - Test: 0.3846 (38.46% of Normal cases correctly identified)
   - This is much lower than recall (Pneumonia class performance)

2. **False Positive Rate**:
   - 144 Normal cases were incorrectly predicted as Pneumonia (out of 234 total Normal cases)
   - This represents 61.54% of Normal cases being misclassified
   - Only 38.46% of Normal cases were correctly identified

3. **False Negative Rate**:
   - Only 2 Pneumonia cases were missed (out of 390 total Pneumonia cases)
   - This represents only 0.51% of Pneumonia cases being misclassified
   - 99.49% of Pneumonia cases were correctly identified

**Why Normal was harder to classify**:

1. **Class Imbalance**: Training set had 74.21% Pneumonia cases, so the model saw more Pneumonia examples
2. **Model Bias**: The weighted loss helped, but the model still learned to favor Pneumonia predictions
3. **Feature Similarity**: At 28×28 resolution, some Normal cases may have features that resemble Pneumonia
4. **Conservative Approach**: The model errs on the side of caution, predicting Pneumonia when uncertain (better for medical diagnosis)

**Implications**:
- The model is highly sensitive to Pneumonia (low false negatives) - good for screening
- Higher false positive rate means more Normal cases need follow-up, but this is acceptable in medical screening
- The high recall (99.49%) ensures almost no Pneumonia cases are missed

---

## 6. Discussion

### 6.1 Challenges Encountered

1. **Class Imbalance**: 
   - The dataset has significant class imbalance (2.88:1 ratio, Pneumonia:Normal)
   - **Impact**: Model tended to favor predicting the majority class (Pneumonia)
   - **Solution**: Implemented weighted binary cross-entropy loss with class weights (Normal=1.9390, Pneumonia=0.6737)
   - **Result**: Successfully improved recall for both classes, though Normal class still more challenging

2. **Test Set Distribution Mismatch**: 
   - Test set has different class distribution (37.5% Normal, 62.5% Pneumonia) vs training/validation (25.8% Normal, 74.2% Pneumonia)
   - **Impact**: Test accuracy (76.60%) lower than validation accuracy (87.79%)
   - **Solution**: Documented the difference and interpreted results accordingly
   - **Result**: Model still performs reasonably on test set, but performance gap indicates distribution shift

3. **Normal Class Classification Difficulty**:
   - Normal class had much lower specificity (38.46% on test set) compared to Pneumonia recall (99.49%)
   - **Impact**: High false positive rate - many Normal cases predicted as Pneumonia
   - **Solution**: Weighted loss helped, but Normal class remains more challenging
   - **Result**: Model is conservative (predicts Pneumonia when uncertain), which is acceptable for medical screening

4. **Low Resolution Images**:
   - 28×28 resolution limits detail visibility
   - **Impact**: Fine-grained features that radiologists use are not clearly visible
   - **Solution**: Used texture (Emboss) and edge (Sobel) kernels to extract relevant features
   - **Result**: Model learned to use spatial patterns despite low resolution

5. **NumPy Implementation Complexity**:
   - Implementing all operations from scratch in NumPy required careful dimension tracking
   - **Impact**: Debugging gradient shapes and ensuring correct matrix operations
   - **Solution**: Systematic testing of each layer and careful dimension verification
   - **Result**: Successfully implemented complete forward and backward passes

6. **Training Time**:
   - Pure NumPy implementation is slower than optimized frameworks
   - **Impact**: Training took longer than with GPU-accelerated frameworks
   - **Solution**: Used batch processing and efficient NumPy operations
   - **Result**: Training completed successfully in reasonable time (~5-10 minutes)

### 6.2 Architecture Choices

The architecture design involved several key decisions, each with specific rationale:

**Why Emboss Kernel for First Layer?**

The Emboss kernel was chosen for the first convolutional layer to enhance texture and depth perception in X-ray images. This choice is motivated by:

1. **Texture Enhancement**: Pneumonia often manifests as subtle variations in tissue density. The emboss filter amplifies these variations, making them more detectable.
2. **Directional Sensitivity**: The asymmetric emboss pattern captures directional texture information, which can help identify irregular patterns associated with pathology.
3. **Feature Priming**: By enhancing texture first, the subsequent layers can work with more pronounced features, improving overall feature extraction.

**Why Sobel Kernel for Second Layer?**

The Sobel kernel (specifically Sobel X for horizontal edge detection) was selected for the second layer because:

1. **Edge Detection**: Pneumonia often presents as areas with distinct boundaries (consolidation, infiltrates). Edge detection helps identify these structural changes.
2. **Complementary to Emboss**: After texture enhancement, edge detection helps identify the boundaries of pathological regions.
3. **Structural Features**: Chest X-rays contain many structural elements (ribs, lung borders, heart silhouette). Edge detection helps the model focus on relevant boundaries.

**Why 16 Units in Dense Layer?**

The choice of 16 units in the first dense layer represents a balance between:

1. **Model Capacity**: 16 units provide sufficient capacity to learn complex feature combinations from the 25 flattened features (5×5 from final pooling).
2. **Overfitting Prevention**: A smaller dense layer (compared to common choices like 64 or 128) reduces the risk of overfitting, especially important given the relatively small dataset (4,708 training samples).
3. **Computational Efficiency**: Fewer parameters mean faster training and inference, which is beneficial for a NumPy-based implementation.
4. **Empirical Balance**: 16 units is large enough to capture non-linear relationships but small enough to maintain generalization.

**Why Dropout Rate of 0.5?**

A dropout rate of 0.5 (50% of neurons randomly deactivated) was chosen because:

1. **Standard Practice**: 0.5 is a commonly used dropout rate that provides good regularization without being too aggressive.
2. **Regularization Balance**: With a small dense layer (16 units), 0.5 dropout provides meaningful regularization without completely disabling the layer.
3. **Prevents Co-adaptation**: By randomly deactivating half the neurons, the model learns more robust features that don't rely on specific neuron combinations.
4. **Empirical Effectiveness**: 0.5 dropout has been shown to work well across many tasks and architectures.

**Additional Architecture Decisions:**

- **No Padding in Convolutions**: Chosen to progressively reduce dimensions, reducing computational cost and parameters. The dimension reduction (28→26→13→11→5) is intentional for efficiency.
- **Two Pooling Layers**: Two max pooling operations reduce spatial dimensions by factor of 4 total, significantly reducing parameters while preserving important features through max selection.
- **Fixed Kernels vs. Learnable**: Fixed kernels (Emboss, Sobel) simplify implementation and provide interpretable features, though learnable kernels might achieve better performance.
- **Single Output Unit**: Binary classification requires only one output unit with sigmoid activation, producing probability scores directly.

### 6.3 Limitations

1. **Low Resolution**:
   - 28×28 images limit detail visibility compared to full-resolution X-rays
   - Fine-grained features (subtle infiltrates, vessel patterns) are not clearly visible
   - May miss subtle pathological signs that radiologists can detect

2. **Simple Architecture**:
   - Only 2 convolutional layers with fixed kernels
   - Limited capacity compared to modern deep networks
   - Fixed kernels (Emboss, Sobel) may not be optimal for this specific task
   - Only 16 units in dense layer limits feature combination capacity

3. **Limited Dataset Size**:
   - 4,708 training samples may be insufficient for complex pattern learning
   - Small validation set (524 samples) may not fully represent generalization
   - Test set (624 samples) provides limited statistical confidence

4. **Fixed Kernels**:
   - Emboss and Sobel kernels are not learnable
   - May not capture optimal features for this specific dataset
   - Learnable kernels could potentially improve performance

5. **Basic Optimizer**:
   - SGD without momentum or adaptive learning rates
   - May converge slower or get stuck in suboptimal minima
   - Advanced optimizers (Adam, RMSprop) could improve training

6. **No Data Augmentation**:
   - Training only on original images
   - No rotation, translation, or brightness variations
   - Could improve generalization and robustness

- Low resolution (28×28) limits detail visibility
- Simple architecture may not capture complex patterns
- Limited dataset size

### 6.4 Suggestions for Improvement

1. **Data Augmentation**:
   - Rotations (±15 degrees), translations, brightness/contrast adjustments
   - Would increase effective dataset size and improve generalization
   - Particularly helpful for Normal class which is harder to classify

2. **Learnable Convolution Kernels**:
   - Replace fixed Emboss/Sobel kernels with learnable convolutional filters
   - Allow model to discover optimal feature extractors for this specific task
   - Could significantly improve feature extraction

3. **Deeper Architecture**:
   - Add more convolutional layers (3-4 layers instead of 2)
   - Use residual connections to enable deeper networks
   - Increase dense layer capacity (32-64 units instead of 16)
   - Would capture more complex hierarchical features

4. **Higher Resolution Images**:
   - Use full-resolution chest X-rays (e.g., 224×224 or 512×512)
   - Would allow detection of fine-grained features
   - Require more computational resources but could improve accuracy

5. **Advanced Optimizers**:
   - Replace SGD with Adam or RMSprop
   - Adaptive learning rates could improve convergence
   - Momentum could help escape local minima

6. **Learning Rate Scheduling**:
   - Implement learning rate decay (reduce LR when validation loss plateaus)
   - Could improve fine-tuning in later epochs
   - Help achieve better convergence

7. **Transfer Learning**:
   - Use pre-trained features from medical imaging datasets
   - Initialize with weights from models trained on similar medical images
   - Fine-tune on PneumoniaMNIST dataset

8. **Ensemble Methods**:
   - Train multiple models with different initializations
   - Combine predictions (voting or averaging)
   - Could improve robustness and reduce variance

9. **Class Balancing Techniques**:
   - In addition to weighted loss, use oversampling (SMOTE) or undersampling
   - Could further improve Normal class classification
   - Might help address the specificity issue

10. **Attention Mechanisms**:
    - Add attention layers to focus on relevant image regions
    - Could help model identify key pathological areas
    - Particularly useful for low-resolution images

11. **Regularization Improvements**:
    - Add L2 weight regularization
    - Use batch normalization
    - Could improve generalization and reduce overfitting

12. **Hyperparameter Tuning**:
    - Systematically tune learning rate, batch size, dropout rate
    - Use cross-validation or validation set for hyperparameter selection
    - Could find better configurations

---

## 7. Conclusion

### 7.1 Summary of Achievements

This project successfully implemented a complete binary classification pipeline for pneumonia detection in chest X-ray images using only NumPy, without high-level deep learning frameworks. The implementation included:

- **Complete CNN Architecture**: Custom-built network with convolution, pooling, dense layers, dropout, and activation functions
- **Full Training Pipeline**: Forward propagation, backward propagation, loss computation, and parameter updates
- **Training Infrastructure**: Batch processing, early stopping, metrics tracking, and model checkpointing
- **Comprehensive Evaluation**: Metrics computation, confusion matrix analysis, and ROC curve generation

### 7.2 Key Findings

1. **Model Performance**:
   - Achieved 87.79% accuracy on validation set and 76.60% on test set
   - High recall (99.49% on test) ensures almost no Pneumonia cases are missed
   - AUC of 0.9051 indicates good class separation ability

2. **Class Imbalance Impact**:
   - Normal class was significantly harder to classify (38.46% specificity vs 99.49% recall for Pneumonia)
   - Weighted loss helped but did not fully resolve the imbalance
   - Model adopts conservative approach, predicting Pneumonia when uncertain

3. **Architecture Effectiveness**:
   - Simple 2-layer CNN with fixed kernels achieved reasonable performance
   - Emboss and Sobel kernels successfully extracted relevant features
   - Model learned to distinguish classes despite low resolution (28×28)

4. **Training Dynamics**:
   - Model converged steadily over 50 epochs
   - No overfitting observed (validation loss tracked training loss)
   - Best model found at epoch 48

### 7.3 Future Work

1. **Immediate Improvements**:
   - Implement learnable convolution kernels
   - Add data augmentation to improve generalization
   - Experiment with deeper architectures

2. **Advanced Techniques**:
   - Apply transfer learning from medical imaging datasets
   - Implement ensemble methods for robustness
   - Use higher resolution images for better feature detection

3. **Clinical Integration**:
   - Validate on larger, more diverse datasets
   - Test on real-world clinical images
   - Develop confidence scoring for predictions
   - Integrate with clinical decision support systems

4. **Technical Enhancements**:
   - Optimize NumPy implementation for faster training
   - Implement advanced optimizers (Adam, RMSprop)
   - Add batch normalization and other regularization techniques

### 7.4 Final Remarks

The project successfully demonstrates that a custom-built neural network in NumPy can achieve reasonable performance on medical image classification, despite limitations in resolution and architecture complexity. The high recall (99.49%) makes the model suitable for screening applications where missing Pneumonia cases is critical. While the Normal class classification needs improvement, the model's conservative approach (high sensitivity) is appropriate for medical diagnosis scenarios.

The implementation provides valuable insights into the inner workings of neural networks and demonstrates that understanding the fundamentals (forward/backward propagation, gradient computation) is essential for effective deep learning practice.

---

## References

1. MedMNIST: https://github.com/MedMNIST/MedMNIST
2. [Add other references as needed]

---

## Appendix

### A. Code Structure
[Brief description of code organization]

### B. Hyperparameters
[Complete list of hyperparameters used]

### C. Additional Visualizations
[Any additional plots or analysis]

