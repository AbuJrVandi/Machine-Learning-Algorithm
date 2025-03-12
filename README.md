# Machine Learning Projects: Logistic Regression and Iris Dataset Classification
This repository contains two comprehensive machine learning workflows:

# Logistic Regression Model: A systematic approach to training and evaluating a logistic regression model on a dataset.
Iris Dataset Classification: A comparison of Random Forest and Decision Tree models for classifying the Iris dataset.
Both projects include data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results. The trained models are saved for future deployment.

# Table of Contents

Logistic Regression Model
Overview
Setup and Imports
Dataset Loading and Exploration
Data Preprocessing
Model Training and Hyperparameter Tuning
Model Evaluation
Model Persistence
Conclusion
Iris Dataset Classification
Project Overview
Objectives
Dataset Description
Installation
Usage
Visualizations
Results and Comparison
Future Work
How to Use
License
Logistic Regression Model

# Overview

This project demonstrates a systematic approach to training and evaluating a logistic regression model. The workflow includes:

Data preprocessing and exploration
Feature scaling and train-test split
Hyperparameter tuning using GridSearchCV
Model evaluation with various metrics (accuracy, precision, recall, F1-score, ROC-AUC, etc.)
Visualization of results (confusion matrix, ROC curve, precision-recall curve, etc.)
Model persistence using joblib for future deployment

# Setup and Imports

## Required Libraries

pandas: For data manipulation and analysis.
numpy: For numerical computations.
matplotlib.pyplot: For creating visualizations.
seaborn: For statistical graphics.
sklearn.model_selection: For data splitting, cross-validation, and hyperparameter tuning.
sklearn.preprocessing: For feature scaling.
sklearn.linear_model: For logistic regression.
sklearn.metrics: For model evaluation.
joblib: For model persistence.
logging: For monitoring script execution.

## Logging Configuration

Logging is configured to capture important steps, errors, and events in model_training.log. This helps in debugging and tracking the progress of the workflow.

# Dataset Loading and Exploration

## Load the Dataset

The dataset is loaded using pandas.read_csv() and basic metadata is displayed using df.info().

Check for Missing Values

Missing values are checked using df.isnull().sum() and handled appropriately.

Split Features and Target

The dataset is split into features (X) and target (y) for model training.

Encoding Categorical Variables

Categorical variables are encoded using pd.get_dummies() to convert them into numerical format.

# Data Preprocessing

Train-Test Split

The dataset is split into training and testing sets using train_test_split() with an 80-20 split ratio.

Feature Scaling

Features are standardized using StandardScaler to ensure they are on a comparable scale.

Model Training and Hyperparameter Tuning

Hyperparameter Tuning with GridSearchCV

A hyperparameter grid is defined, and GridSearchCV is used to find the best combination of parameters using 5-fold cross-validation.

Model Training

The logistic regression model is trained using the best hyperparameters found by GridSearchCV.

Model Evaluation

Model Prediction

Predictions are made using the trained model on the test dataset.

Accuracy and Classification Report

The accuracy of the model is calculated using accuracy_score(), and a classification report is generated with classification_report().

Confusion Matrix

The confusion matrix is plotted using seaborn.heatmap() to visualize the model’s performance.

ROC Curve and AUC

The ROC curve is plotted to assess the trade-off between the true positive rate and false positive rate, and AUC is calculated.

Precision-Recall Curve

The precision-recall curve is plotted to assess the model's ability to discriminate between classes.

Feature Importance

Feature importance is calculated from the coefficients of the logistic regression model and visualized as a bar chart.

# Model Persistence

Saving the Trained Model

The trained model is saved using joblib.dump() for future use.

Loading the Saved Model

The saved model can be loaded using joblib.load() for making predictions on new data.

Conclusion

This project implements a robust approach to training, evaluating, and optimizing a logistic regression model. It includes key machine learning processes such as data preprocessing, hyperparameter tuning, cross-validation, and performance evaluation. The trained model is saved for future deployment, ensuring reusability and scalability in real-world applications.

#Iris Dataset Classification

Project Overview

This project applies machine learning techniques to classify the Iris dataset using two popular models: Random Forest and Decision Tree. The primary objective is to compare the performance of these models on classification tasks and analyze their strengths and weaknesses.

Key aspects of the workflow include:

Data Preprocessing: Handling missing values, feature scaling, and encoding categorical variables.
Model Training: Training Random Forest and Decision Tree classifiers.
Model Evaluation: Using metrics like accuracy, classification report, confusion matrix, and feature importance.
Hyperparameter Tuning: Optimizing models using GridSearchCV.
Cross-Validation & Learning Curves: Ensuring model stability and diagnosing overfitting/underfitting.

# Objectives

Load and explore the Iris dataset.
Preprocess data by handling missing values and encoding categorical variables.
Train and evaluate Random Forest and Decision Tree models.
Compare model performance using accuracy, confusion matrix, and feature importance.
Perform hyperparameter tuning with GridSearchCV.
Implement cross-validation and learning curve analysis for further model evaluation.

# Dataset Description

The dataset consists of 150 samples of iris flowers, each described by 4 numerical features:

Sepal Length
Sepal Width
Petal Length
Petal Width
The target variable is the species of the iris flower, with 3 classes:

Setosa
Versicolor
Virginica
The dataset is balanced, with each class containing 50 samples.

# Installation

To run this project locally, follow these steps:

Clone the repository:
bash
Copy
git clone https://github.com/yourusername/iris-classification.git
Navigate into the project directory:
bash
Copy
cd iris-classification
Install the required dependencies:
bash
Copy
pip install -r requirements.txt

# Usage

Data Preprocessing: The dataset is loaded, cleaned, and preprocessed. Missing values (if any) are handled, and categorical variables are encoded for model compatibility.
Model Training and Evaluation:
Both Random Forest and Decision Tree models are trained.
The models are evaluated using accuracy, classification report, and confusion matrix.
Hyperparameter Tuning: GridSearchCV is used to fine-tune the hyperparameters of the Random Forest model to achieve optimal performance.
Cross-Validation: Stratified K-Fold cross-validation is implemented to ensure stable model evaluation.
Learning Curve Analysis: Learning curves are generated to visualize the model's performance with varying training data sizes.

# Visualizations

Confusion Matrix Heatmap: Visualizes misclassifications across different classes.
Feature Importance Bar Plot: Highlights the most influential features for each model.
Decision Tree Visualization: Displays the decision-making process of the Decision Tree model.
Precision-Recall Curve: Assesses precision and recall trade-offs for binary classification.
Learning Curves: Diagnose overfitting or underfitting by analyzing model performance with different data sizes.

# Results and Comparison

The Random Forest model provides a more robust and reliable classification performance compared to the Decision Tree.
Feature importance and confusion matrix visualizations help understand model behavior and decision-making.
Future Work

Further hyperparameter optimization to enhance model performance.
Explore other ensemble techniques like Gradient Boosting and AdaBoost.
Investigate the impact of additional features or feature engineering techniques.
How to Use

# Clone the repository.
Install the required libraries using pip install -r requirements.txt.
Run the script to train and evaluate the logistic regression model or the Iris dataset classification models.
Check the model_training.log file for execution logs.
Use the saved models (logistic_regression_model.pkl, random_forest_model.pkl, etc.) for making predictions on new data.

# License

This project is licensed under the MIT License. See the LICENSE file for details.
