# PneumoniaMNIST Binary Classification Report

**Student:** Mustafa Ocak | 2021400159  

---

## 1. Introduction

This project implements a binary classification pipeline for chest X-ray images to detect pneumonia using a custom-built neural network in NumPy. The dataset is PneumoniaMNIST (28×28 grayscale images) with two classes: Normal (0) and Pneumonia (1).

---

## 2. Data Exploration and Analysis

### 2.1 Class Distribution

**Table 1: Class Distribution Across Splits**

| Split | Normal (0) | Pneumonia (1) | Total | Normal % | Pneumonia % |
|-------|------------|---------------|-------|----------|--------------|
| Training | 1,214 | 3,494 | 4,708 | 25.79% | 74.21% |
| Validation | 135 | 389 | 524 | 25.76% | 74.24% |
| Test | 234 | 390 | 624 | 37.50% | 62.50% |

**Figure 1: Class Distribution Visualization**
<img width="2233" height="594" alt="class_distribution" src="https://github.com/user-attachments/assets/371a3d6d-788a-4bcb-ad6a-e720f1dd5700" />

**Observations:**
- Training and validation sets have similar distributions (~25.8% Normal, ~74.2% Pneumonia)
- Test set has different distribution (37.5% Normal, 62.5% Pneumonia)
- Significant class imbalance (2.88:1 ratio) addressed with weighted loss

### 2.2 Sample Images Visualization

**Figure 2: Sample Images from Each Class**
<img width="1723" height="742" alt="sample_images" src="https://github.com/user-attachments/assets/f6a40eba-e5b8-4ae0-a1c2-6938f11dcabf" />

Shows 5 sample images from each class (Normal on top row, Pneumonia on bottom row).

### 2.3 Pixel Intensity Histograms

**Figure 3: Pixel Intensity Histograms**
<img width="1783" height="593" alt="pixel_histograms" src="https://github.com/user-attachments/assets/5c00db76-c0a8-4ffe-be8a-87a90fcf17d6" />


**Statistics:**
- Normal Class: Mean=139.90, Std=46.82
- Pneumonia Class: Mean=147.90, Std=41.29

**Findings:**
- Significant overlap between classes indicates spatial patterns are needed, not just intensity values
- Slight difference in mean intensity (Pneumonia higher) but overlap is substantial

### 2.4 Data Preprocessing

Images normalized to [0, 1] by dividing by 255.0. Class weights calculated for weighted loss:
- Weight for Normal (0): 1.9390
- Weight for Pneumonia (1): 0.6737

---

## 3. Network Architecture

### 3.1 Architecture Overview

The network consists of 8 layers:

**Feature Extraction:**
1. Conv1: 3×3 Emboss kernel → (N, 26, 26)
2. MaxPool1: 2×2, stride=2 → (N, 13, 13)
3. Conv2: 3×3 Sobel kernel → (N, 11, 11)
4. MaxPool2: 2×2, stride=2 → (N, 5, 5)

**Classification:**
5. Flatten → (N, 25)
6. Dense1: 16 units + ReLU → (N, 16)
7. Dropout: rate=0.5 → (N, 16)
8. Dense2: 1 unit + Sigmoid → (N, 1)

### 3.2 Dimension Flow

**Table 2: Network Dimension Flow**

| Layer | Input Shape | Output Shape | Transformation |
|-------|-------------|--------------|---------------|
| Input | (N, 28, 28) | (N, 28, 28) | - |
| Conv1 (Emboss 3×3) | (N, 28, 28) | (N, 26, 26) | Convolution, no padding |
| MaxPool1 (2×2) | (N, 26, 26) | (N, 13, 13) | Max pooling, stride=2 |
| Conv2 (Sobel 3×3) | (N, 13, 13) | (N, 11, 11) | Convolution, no padding |
| MaxPool2 (2×2) | (N, 11, 11) | (N, 5, 5) | Max pooling, stride=2 |
| Flatten | (N, 5, 5) | (N, 25) | Reshape |
| Dense1 (16 units) | (N, 25) | (N, 16) | Linear + ReLU |
| Dropout (0.5) | (N, 16) | (N, 16) | Mask (training) |
| Dense2 (1 unit) | (N, 16) | (N, 1) | Linear + Sigmoid |

**Total Trainable Parameters:** 433 (Dense1: 416, Dense2: 17)

### 3.3 Kernel Choices and Function Roles

**Emboss Kernel (Layer 1):**
- Purpose: Texture enhancement and depth perception
- Helps emphasize subtle density variations in X-ray images

**Sobel Kernel (Layer 2):**
- Purpose: Edge detection and boundary identification
- Detects horizontal edges to identify structural abnormalities

**Activation Functions:**
- ReLU: Used in Dense1 for non-linearity and gradient flow
- Sigmoid: Used in output layer to produce probability scores [0, 1]

---

## 4. Training Procedure

### 4.1 Training Configuration

- **Batch Size:** 32
- **Learning Rate:** 0.01
- **Max Epochs:** 50
- **Early Stopping Patience:** 5
- **Loss Function:** Weighted Binary Cross-Entropy
- **Optimizer:** SGD

### 4.2 Training and Validation Loss

**Figure 4: Training and Validation Loss Over Epochs**
<img width="1483" height="882" alt="training_loss" src="https://github.com/user-attachments/assets/6d08f424-ff66-49d7-bdb7-e224b91a0a22" />


Training completed for 50 epochs:
- Training loss decreased from 0.8036 to 0.2825
- Validation loss decreased from 0.4825 to 0.2453
- Best epoch: 48 (validation loss: 0.2376)
- Steady convergence with no overfitting

### 4.3 Validation Metrics Over Epochs

**Figure 5: Validation Metrics Over Epochs**
<img width="2084" height="1476" alt="validation_metrics" src="https://github.com/user-attachments/assets/b64b6dbb-4d9a-43e0-ab2c-bb80f825eebf" />


Metrics improved consistently:
- Accuracy: 0.7424 → 0.8779
- Precision: 0.8521 → 0.8719
- Recall: 0.8521 → 0.9794
- F1 Score: 0.8521 → 0.9225

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

**Key Findings:**
- High recall (99.49% on test) ensures almost no Pneumonia cases are missed
- Lower specificity (38.46% on test) indicates difficulty classifying Normal cases
- Test accuracy (76.60%) lower than validation (87.79%), possibly due to distribution mismatch

### 5.2 Confusion Matrix

**Figure 6: Confusion Matrix - Test Set**
<img width="1111" height="882" alt="confusion_matrix_test" src="https://github.com/user-attachments/assets/8fcfacd2-6ed1-4129-addb-cba8230caaf9" />


**Test Set Confusion Matrix:**

|                | Predicted Normal | Predicted Pneumonia |
|----------------|------------------|---------------------|
| **Actual Normal** | 90 (TN) | 144 (FP) |
| **Actual Pneumonia** | 2 (FN) | 388 (TP) |

**Analysis:**
- Very low false negatives (2) - excellent for medical diagnosis
- Higher false positives (144) - model tends to predict Pneumonia when uncertain

### 5.3 ROC Curve

**Figure 7: ROC Curve - Test Set**
<img width="1183" height="882" alt="roc_curve_test" src="https://github.com/user-attachments/assets/682f3cd5-1fe0-47da-bd7d-97e6a6464743" />


- **AUC Score:** 0.9051 - indicates good ability to distinguish between classes
- Curve shows strong performance with high true positive rate at low false positive rates

### 5.4 Per-Class Analysis

**Which class was harder to classify?**

**Answer: The Normal class was significantly harder to classify than the Pneumonia class.**

**Evidence:**
1. **Specificity (Normal class):** 38.46% on test set vs **Recall (Pneumonia class):** 99.49%
2. **False Positive Rate:** 144 Normal cases incorrectly predicted as Pneumonia (61.54% of Normal cases)
3. **False Negative Rate:** Only 2 Pneumonia cases missed (0.51% of Pneumonia cases)

**Reasons:**
1. Class imbalance: Training set had 74.21% Pneumonia cases
2. Model bias: Learned to favor Pneumonia predictions
3. Feature similarity: At 28×28 resolution, some Normal cases resemble Pneumonia
4. Conservative approach: Model predicts Pneumonia when uncertain (appropriate for medical screening)

---

## 6. Discussion

### 6.1 Challenges Encountered

1. **Class Imbalance:** Significant imbalance (2.88:1 ratio) addressed with weighted loss, but Normal class remains more challenging
2. **Test Set Distribution Mismatch:** Different class distribution (37.5% Normal vs 25.8% in train/val) may contribute to performance gap
3. **Normal Class Difficulty:** Lower specificity (38.46%) indicates high false positive rate
4. **Low Resolution:** 28×28 images limit detail visibility
5. **NumPy Implementation:** Required careful dimension tracking and gradient verification

### 6.2 Suggestions for Improvement

1. **Data Augmentation:** Rotations, translations, brightness adjustments to increase effective dataset size
2. **Learnable Kernels:** Replace fixed Emboss/Sobel kernels with learnable convolutional filters
3. **Deeper Architecture:** Add more convolutional layers and increase dense layer capacity
4. **Higher Resolution:** Use full-resolution X-rays (224×224 or 512×512) for better feature detection
5. **Learning Rate Scheduling:** Implement LR decay when validation loss plateaus
6. **Transfer Learning:** Use pre-trained features from medical imaging datasets
7. **Regularization:** Add L2 weight regularization and batch normalization
8. **Hyperparameter Tuning:** Systematically tune learning rate, batch size, dropout rate

---

## 7. Conclusion

The model achieved 76.60% accuracy on the test set with 99.49% recall, ensuring almost no Pneumonia cases are missed. The Normal class was more challenging to classify (38.46% specificity), but the model's conservative approach (high sensitivity) is appropriate for medical screening applications.

---

## References

1. MedMNIST: https://github.com/MedMNIST/MedMNIST
