# Breast-Cancer-Prediction-using-ML-Algorithms

#### **About Dataset**
🦠 Breast Cancer Data Set
This dataset contains the characteristics of patients diagnosed with cancer. The dataset contains a unique ID for each patient, the type of cancer (diagnosis), the visual characteristics of the cancer and the average values of these characteristics.

📚 The main features of the dataset are as follows:
* **id:** Represents a unique ID of each patient.
* **diagnosis:** Indicates the type of cancer. This property can take the values "M" (Malignant - Benign) or "B" (Benign - Malignant).
* **radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean:** Represents the mean values of the cancer's visual characteristics.

* ## There are also several categorical features where patients in the dataset are labeled with numerical values.
* ## Each sample contains the patient's unique ID, the cancer diagnosis and the average values of the cancer's visual characteristics.

## This repository contains several machine learning projects focused on diagnosing cancer as malignant or benign based on visual features. Below are the detailed descriptions of the projects utilizing different algorithms:

1. **Logistic Regression**
Logistic Regression is a straightforward and interpretable algorithm used for binary classification problems. Given the binary nature of the dataset (malignant or benign tumors), logistic regression is well-suited for this task. The algorithm models the probability that a given input point belongs to a particular class using a logistic function.

* Example Project:

* Cancer Prediction Using Logistic Regression:
  * Develop a logistic regression model to predict the type of tumor (malignant or benign) based on visual features.
  * Fit the logistic regression model to the training data, optimize it, and evaluate its performance using accuracy, precision, recall, and F1-score.
  * The simplicity and interpretability of logistic regression make it a strong baseline model for this classification task.

2. **Support Vector Machines (SVM)**
Support Vector Machines are powerful classification algorithms that aim to find the optimal hyperplane that separates classes in the feature space. SVM is particularly effective for high-dimensional spaces and cases where the classes are well-separated.

* Example Project:

* Cancer Classification Using SVM:
  * Apply the SVM algorithm to classify tumors as malignant or benign.
* Tune hyperparameters such as the kernel type and regularization parameter.
* SVM's ability to create a clear margin of separation between classes makes it a strong candidate for this classification task, especially when dealing with complex and high-dimensional data.

3. **Decision Tree**
Decision Trees are a simple yet powerful method for classification tasks. They split the data based on feature values, creating a tree-like model of decisions. Each node represents a decision rule, and each leaf node represents a class label.

* Example Project:

* Cancer Diagnosis Using Decision Tree:
  * Develop a decision tree model to predict whether a tumor is malignant or benign.
  * Grow the tree using the training data, prune to prevent overfitting, and interpret the resulting decision rules.
  * Decision trees are highly interpretable and can provide clear insights into the factors influencing the diagnosis.

4. **Random Forest**
Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and robustness. It reduces the risk of overfitting compared to a single decision tree and handles variability and noise effectively.

* Example Project:

* Cancer Classification Using Random Forest:
  * Implement a random forest model to classify tumors as malignant or benign.
  * Train multiple decision trees on different subsets of the data and combine their predictions.
  * The ensemble approach of random forests enhances model performance and provides a robust and reliable tool for cancer diagnosis.

## General Steps for Each Project:

### Data Preprocessing:

* Load and clean the dataset.
* Handle missing values and outliers.
* Normalize or standardize the features if necessary.
* Split the data into training and testing sets.

### Model Training:

* Choose appropriate hyperparameters for each algorithm.
* Train the model on the training data.
* Perform cross-validation to fine-tune hyperparameters and assess model stability.

### Model Evaluation:

* Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
* Compare the results of different models to determine the best-performing algorithm.

### Model Interpretation and Validation:

* Interpret the model's results and decision boundaries.
* Validate the model using a hold-out test set or through cross-validation.
