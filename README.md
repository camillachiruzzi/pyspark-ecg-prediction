# Distributed ECG Analysis and Heart Attack Prediction with PySpark

This project applies distributed data analysis and machine learning techniques to a medical dataset of ECG recordings. The objective is to extract meaningful features and **predict ischemic heart disease (heart attacks)** using scalable tools such as **Apache Spark**.

---

## Dataset

This project uses the publicly available [Ischemia ECG Dataset](https://www.kaggle.com/datasets/bjoernjostein/ischemia-dataset) from Kaggle.

- Contains over **2,500 ECG recordings** in PhysioNet-compatible `.mat` and `.hea` formats.
- Labeled for **ischemic heart disease** (including heart attack events).
- Data used for both feature extraction and supervised classification.

---

## Overview

- **Goal**: Binary classification – predict whether a patient has suffered a heart attack  
- **Key technologies**: PySpark, NeuroKit2, Scikit-learn  
- **Focus**: Feature engineering on biomedical signals, distributed processing, model evaluation

---

## Project Structure

### `1_dataframe_creation.ipynb`
- Parses and merges `.hea` and `.mat` files into a unified PySpark DataFrame
- Uses custom dictionaries and partition tuning to handle large arrays
- Cleans constant or empty fields and filters for consistent signal lengths

### `2_data_preparation_understanding.ipynb`
- Converts signal derivations into numerical arrays
- Engineers statistical features (mean, std) from ECG leads grouped by anatomical region
- Extracts heart rate features using **NeuroKit2** (e.g., BPM, R-peak detection)
- Produces visualizations: ECG waves, age distributions, average BPM, correlation matrix

### `3_classification.ipynb`
- Defines target variable from diagnostic codes (heart attack: yes/no)
- Trains two models:
  - **Random Forest** (with Grid Search, 5-fold CV, feature importance)
  - **Multilayer Perceptron (MLP)** with tuned layer architecture
- Handles class imbalance via oversampling
- **Best model (Random Forest on balanced data)**:
  - Accuracy: 0.93  
  - AUC: 0.97  
  - Precision: 0.92  
  - Recall: 0.98

---
