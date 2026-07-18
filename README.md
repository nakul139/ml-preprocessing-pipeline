# Machine Learning Data Preprocessing Pipeline

This repository contains my practice implementation of a machine learning data preprocessing pipeline using **Scikit-learn**. The goal is to prepare the California Housing dataset for training machine learning models by applying industry-standard preprocessing techniques.

## 📌 Project Overview

The project demonstrates how to:

- Load and inspect a dataset
- Split data using **StratifiedShuffleSplit**
- Handle missing values with **SimpleImputer**
- Scale numerical features using **StandardScaler**
- Encode categorical features using **OneHotEncoder**
- Combine preprocessing steps using **Pipeline**
- Apply transformations using **ColumnTransformer**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## 📂 Project Structure

```
.
├── housing.csv
├── preprocessing_pipeline.py
├── requirements.txt
└── README.md
```

## ⚙️ Workflow

1. Load the California Housing dataset.
2. Create income categories for stratified sampling.
3. Split the dataset into training and testing sets.
4. Separate features and target variable.
5. Build preprocessing pipelines for:
   - Numerical features
   - Categorical features
6. Combine pipelines using `ColumnTransformer`.
7. Fit and transform the training dataset.

## 📚 Concepts Practiced

- Data Preprocessing
- Stratified Sampling
- Feature Scaling
- Missing Value Imputation
- One-Hot Encoding
- Scikit-learn Pipelines
- ColumnTransformer

## 🚀 Future Improvements

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Train Machine Learning Models
- Model Evaluation
- Hyperparameter Tuning

## 📖 Learning Purpose

This project is created for learning and practicing machine learning preprocessing techniques using Scikit-learn.
