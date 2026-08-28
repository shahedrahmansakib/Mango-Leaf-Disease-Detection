# Mango Leaf Disease Detection & Classification Pipeline

An end-to-end deep learning framework designed for automated detection and 8-class categorization of mango leaf diseases. This project evaluates and compares two distinct architectural approaches: a **Custom Deep CNN** built from scratch and a fine-tuned **ResNet50 Transfer Learning** model. Built using TensorFlow/Keras and designed for real-time agricultural diagnostics.

---

## 📊 Dataset Specifications
- **Total Dataset Size:** 4,000 high-resolution leaf images
- **Number of Classes:** 8 Categories
  - Anthracnose
  - Bacterial Canker
  - Cutting Weevil
  - Die Back
  - Gall Midge
  - Healthy
  - Powdery Mildew
  - Sooty Mould
- **Dataset Partitioning (Stratified Split):**
  - **Training Set (70%):** 2,800 images
  - **Validation Set (15%):** 600 images
  - **Test Set (15%):** 600 images

---

## 🛠️ Data Preprocessing & Augmentation Pipeline
To enhance model generalization and prevent overfitting, input images are processed through an automated pipeline:
- **Resizing & Scaling:** Standardized to $224 \times 224 \times 3$ pixels and normalized to $[0, 1]$.
- **Augmentation Layers:** Integrated dynamically using TensorFlow Keras preprocessing layers:
  - Random Horizontal & Vertical Flips
  - Random Rotation ($\pm 20\%$)
  - Random Zoom ($\pm 20\%$)
  - Random Contrast Adjustments ($\pm 20\%$)
- **Pipeline Optimization:** Implemented `tf.data` API featuring parallel execution (`AUTOTUNE`), batch processing (`Batch Size = 16`), and prefetching.

---

## 🏗️ Model Architectures

### 1. Custom CNN Architecture
A lightweight deep Convolutional Neural Network built from scratch:
- **Feature Extractor:** 4 sequential blocks of Conv2D layers (32, 64, 128, 256 filters with ReLU activation & same padding), paired with `BatchNormalization` and `MaxPooling2D`.
- **Classification Head:** `GlobalAveragePooling2D` followed by a Dense layer (256 units), Dropout ($rate = 0.3$), and Softmax output (8 units).
- **Total Parameters:** ~458,184 (~1.75 MB)

### 2. Fine-Tuned ResNet50 Architecture
A pre-trained deep residual network leveraging transfer learning:
- **Backbone:** ResNet50 initialized with `ImageNet` weights, fully un-frozen (`base_model.trainable = True`) for end-to-end fine-tuning.
- **Classification Head:** Integrated `GlobalAveragePooling2D`, `BatchNormalization`, Dense layer (256 units, ReLU), Dropout ($rate = 0.3$), and 8-class Softmax activation.
- **Total Parameters:** 23,587,712 (~89.98 MB)

---

## ⚙️ Training Setup & Callbacks

| Parameter / Technique | Custom CNN | Fine-Tuned ResNet50 |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | Cosine Decay Scheduler ($5 \times 10^{-5}$) | Fixed Fine-Tuning ($1 \times 10^{-5}$) |
| **LR Reduction Callback** | N/A | `ReduceLROnPlateau` (factor=0.2, patience=4) |
| **Loss Function** | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |
| **Epochs Trained** | 40 | 20 |
| **Memory Optimization** | `tf.keras.backend.clear_session` | `ClearMemoryCallback` (Garbage Collector) |
| **Model Checkpoint** | Save Best (`val_accuracy`) | Save Best (`val_accuracy`) |

---

## 📈 Performance Comparison & Results

Both models achieved high diagnostic accuracy across the 8 plant health categories:

| Model Architecture | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss | Total Params |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | **99.25%** | **99.67%** | 0.0391 | 0.0208 | **~0.45M** |
| **ResNet50** | **99.46%** | **99.17%** | 0.0150 | 0.0312 | **~23.5M** |

---
