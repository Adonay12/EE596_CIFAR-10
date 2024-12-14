# Deep Residual CNN for CIFAR-10 Classification

This project implements a **Deep Residual Convolutional Neural Network (Deep Residual CNN)** to classify images from the CIFAR-10 dataset. The architecture employs advanced deep learning techniques, including residual blocks and learning rate schedulers, to achieve state-of-the-art accuracy in image classification tasks.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion and Insights](#discussion-and-insights)
- [Future Directions](#future-directions)
- [Model Comparison](#model-comparison)

---

## Introduction

The objective of this project is to build a robust image classification model for the CIFAR-10 dataset using **deep residual learning techniques**. Residual blocks are leveraged to mitigate the vanishing gradient problem, enhancing the model's ability to train deep architectures effectively.

### Motivation
- Classify CIFAR-10 images across 10 object categories.
- Leverage advanced neural network architectures to achieve high test accuracy.

---

## Dataset

The CIFAR-10 dataset consists of:
- **60,000 images** of size 32x32, spread across **10 classes**:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **50,000 training images** and **10,000 test images**.

---

## Methodology

### Architecture
- **Deep Residual CNN (ResNet-like)**:
  - Residual blocks with skip connections to improve gradient flow.
  - Four layers with increasing channel depth.
  - Batch normalization and ReLU activation.

### Training
- **Learning Rate Scheduler**: Used `OneCycleLR` for dynamic adjustment of learning rates.
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: SGD with momentum and weight decay.
- **Data Augmentation**:
  - Random cropping and flipping.
  - Normalization and random erasing.

---

## Results

### Performance
- Achieved **test accuracy** of approximately **90-95%** after 100 epochs.
- Consistent decrease in training loss over epochs, with plans to extend training to 200 epochs.

### Highlights
- Effective use of **residual connections** mitigated vanishing gradient issues.
- **OneCycleLR** scheduler enhanced training efficiency.

---

## Discussion and Insights

- **Residual Blocks**: Successfully addressed the challenges of deeper networks by preserving gradient flow.
- **Dynamic Learning Rates**: `OneCycleLR` proved critical for achieving optimal training efficiency.
- The approach highlights the effectiveness of combining advanced architectures with robust optimization strategies.

---

## Future Directions

To further improve the model's performance, the following enhancements can be explored:
1. **Deeper Architectures**: Experiment with more complex residual architectures.
2. **Advanced Data Augmentation**: Introduce additional augmentation techniques to improve generalization.
3. **Transfer Learning**: Leverage pre-trained models for better initialization.
4. **Extended Training**: Train the model for more epochs (e.g., 200) to maximize accuracy.

---

## Model Comparison

| Feature                        | Deep Residual CNN       | Basic CNN                 |
|--------------------------------|-------------------------|---------------------------|
| **Accuracy**                   | 95.13% (100 epochs)     | 70.24% (20 epochs)        |
| **Architecture**               | Residual blocks, batch normalization | Simpler architecture, no residuals |
| **Learning Rate Adjustment**   | `OneCycleLR` dynamic adjustment | Constant learning rate    |
| **Data Augmentation**          | Advanced augmentation   | Basic augmentation        |

---

## Training Visualization

### Training Loss and Accuracy
Training performance consistently improved over 100 epochs. Residual connections and dynamic learning rates contributed significantly to stability and faster convergence.

---

## Acknowledgments

This project was developed as part of the **EE596 - Practical Introduction to Deep Learning Applications and Theory Final Project**.

---

