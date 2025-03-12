## Iris Dataset Classification with Machine Learning

# Project Overview

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
git clone https://github.com/yourusername/iris-classification.git
Navigate into the project directory:
cd iris-classification
Install the required dependencies:
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

# Future Work

Further hyperparameter optimization to enhance model performance.
Explore other ensemble techniques like Gradient Boosting and AdaBoost.
Investigate the impact of additional features or feature engineering techniques.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
