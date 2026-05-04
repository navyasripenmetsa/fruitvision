# FruitVision: Fruit Classification using Traditional Machine Learning


## Overview

This project implements a traditional machine learning pipeline for classifying fruits using the Fruits-360 dataset. The goal is to evaluate how well handcrafted visual features, combined with classical classifiers, perform on a large-scale, fine-grained image classification problem spanning 141 fruit categories.

The project does not use deep learning. All classification is performed using feature engineering and supervised learning algorithms available in scikit-learn, making it a strong reference for understanding the fundamentals of image-based classification without neural networks.

---

## Dataset

**Fruits-360** is a publicly available, well-curated image dataset consisting of 94,110 labeled images across 141 fruit, vegetable, and nut categories. Each image is captured from multiple angles to introduce visual diversity and help models generalize across viewpoints and natural variations in size, shape, and color.

All images are standardized in size and format, and the dataset includes pre-defined training and testing splits for reliable benchmarking.

**Download the Fruits-360 dataset:**  
https://drive.google.com/file/d/1PzPXuGqnyzcEwcwmgyBxtyxnj0f7AknC/view?usp=sharing

---

## Feature Extraction

Two complementary feature extraction techniques were applied to represent each image as a fixed-length numerical vector.

### Color Histogram (24 Features)

Color is one of the most discriminative visual cues for distinguishing fruit categories. For each image, normalized histograms were computed across the Red, Green, and Blue channels using 8 bins per channel. The three channel histograms were concatenated into a single 24-dimensional feature vector capturing the dominant color distribution of each fruit.

**Extracted CSV Files:**

| Split | Link |
|-------|------|
| Training | https://drive.google.com/file/d/1shSwoXCPUEYywl2Pd46Rnni3Bwc7zhpF/view?usp=sharing |
| Testing | https://drive.google.com/file/d/1QHniGVHU-zDWnZFrk9UQIbNCzLaKNkVN/view?usp=sharing |

### Histogram of Oriented Gradients - HOG (324 Features)

HOG captures edge directions and local shape structure, offering information complementary to color. Images were resized to 32x32 pixels, converted to grayscale, divided into 8x8 pixel cells, and gradient histograms were computed and normalized across overlapping blocks. The resulting descriptor is a 324-dimensional vector encoding the structural and textural properties of each fruit.

**Extracted CSV Files:**

| Split | Link |
|-------|------|
| Training | https://drive.google.com/file/d/19c0VUYrAUTLDCoupl0Y6sg101YYKeVQJ/view?usp=sharing |
| Testing | https://drive.google.com/file/d/1uLKTyg0aYnRd1FgSgyKBpzFfjyEMowrB/view?usp=sharing |

---

## Classifiers Evaluated

Five supervised learning algorithms were trained and evaluated independently on both feature sets.

| Classifier | Key Characteristics |
|------------|---------------------|
| K-Nearest Neighbors (KNN) | Instance-based, sensitive to choice of k and feature dimensionality |
| Support Vector Machine (SVM) | Linear kernel, effective in high-dimensional spaces |
| Logistic Regression | Linear probabilistic model, uses L2 regularization and polynomial feature expansion for color histograms |
| Decision Tree | Interpretable, prone to overfitting on high-dimensional features |
| Naive Bayes | Gaussian variant, assumes feature independence |

---

## Results

All models were evaluated on the held-out test split. Accuracy is reported for each classifier and feature type.

| Classifier | Color Histogram Accuracy (%) | HOG Accuracy (%) |
|---|---|---|
| SVM | **97.07** | **87.10** |
| Logistic Regression | 95.45 | 79.17 |
| KNN | 94.94 | 85.34 |
| Decision Tree | 86.75 | 61.00 |
| Naive Bayes | 85.42 | 55.27 |

**Key observations:**

- SVM with a linear kernel achieved the highest accuracy across both feature types, demonstrating its strength in high-dimensional classification tasks.
- Color Histogram features consistently outperformed HOG features across all classifiers. This suggests that fruit color is a more discriminative signal than local shape structure for this dataset.
- KNN and Logistic Regression showed competitive and reliable performance, particularly on color histogram features.
- Decision Tree performance degraded significantly on HOG features, indicating overfitting to training data and poor generalization in high-dimensional spaces.
- Naive Bayes performed the weakest overall, likely because its feature independence assumption is violated by the correlated structure of HOG descriptors.

---

## Project Structure

```
root/
├── Streamlit_UI/
│   └── Intro_UI.py           # Main Streamlit application entry point
├── Backend/
│   ├── Datasets/             # Place the four extracted CSV files here
│   │   ├── colour_Histogram_Training.csv
│   │   ├── colour_Histogram_Testing.csv
│   │   ├── HOG_Training.csv
│   │   └── HOG_Testing.csv
│   └── Models/               # Place all pre-trained model files here
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install streamlit streamlit-option-menu numpy pandas scikit-learn matplotlib
```

### 3. Download and place datasets

Download the four CSV files from the links in the Feature Extraction section above. Place all four files inside:

```
Backend/Datasets/
```

### 4. Download pre-trained models

Download all model files from the following Google Drive folder:  
https://drive.google.com/drive/folders/1F--y6rWHekOlLxou_b0SzIS___Stvg9K?usp=sharing

Create a folder named `models` inside the `Backend/` directory and place all downloaded model files there:

```
Backend/Models/
```

### 5. Run the application

From the project root directory, run:

```bash
streamlit run Streamlit_UI/Intro_UI.py
```
