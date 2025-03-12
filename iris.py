import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 1. Load the CSV file into a DataFrame
data = pd.read_csv('iris.csv')  # Replace 'your_data.csv' with your file path
logger.info("Data loaded successfully!")
logger.info(data.head())

# 2. Initial Exploration
logger.info(data.info())  # Data types and non-null counts
logger.info(data.describe())  # Statistical summary for numerical features

# 3. Check for missing values
missing_values = data.isnull().sum()
logger.info("Missing values per column:\n%s", missing_values)

# 4. Handling missing values (Example: filling numerical with median and categorical with mode)
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)

# 5. Processing categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
logger.info("Categorical columns: %s", categorical_cols)

# Label Encoding for categorical data
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# 6. Splitting the data into features and target variable
X = data.drop('species', axis=1)
y = data['species']

# 7. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Training and test sets created!")

# 8. Scaling numerical features (using Pipeline)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
X_train = pipeline.named_steps['scaler'].fit_transform(X_train)
X_test = pipeline.named_steps['scaler'].transform(X_test)

# 9. Train and evaluate Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_test_pred = rf_model.predict(X_test)

# Accuracy Score
test_accuracy = accuracy_score(y_test, y_test_pred)
logger.info("Random Forest Test Accuracy: %.2f", test_accuracy)

# Classification Report
logger.info("\nClassification Report:\n%s", classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
logger.info("Confusion Matrix:\n%s", cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Random Forest")
plt.show()

# Feature Importance for Random Forest
feature_importances = rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance for Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# 10. Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
logger.info("Confusion Matrix for Decision Tree:\n%s", cm_dt)

# Visualize Confusion Matrix for Decision Tree
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title("Confusion Matrix for Decision Tree")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str), rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Feature Importance for Decision Tree
dt_feature_importances = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=dt_feature_importances, y=feature_names)
plt.title("Feature Importance for Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# 11. Precision-Recall Curve (for binary or multiclass classification)
if len(np.unique(y_test)) == 2:
    precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title("Precision-Recall Curve for Random Forest")
    plt.show()
else:
    logger.info("Precision-Recall Curve is only applicable for binary classification problems.")

# 12. Hyperparameter Tuning using GridSearchCV (for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
logger.info("Best parameters for Random Forest: %s", grid_search.best_params_)

# 13. Cross-Validation with Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
logger.info("Cross-Validation Scores: %s", cv_scores)
logger.info("Mean CV Score: %.2f", cv_scores.mean())

# 14. Model Comparison (Random Forest vs Decision Tree)
models = {
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("%s Test Accuracy: %.2f", name, accuracy)

# 15. Learning Curves (to diagnose overfitting/underfitting)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score')
plt.plot(train_sizes, test_scores_mean, label='Cross-Validation Score')
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.title("Learning Curves for Random Forest")
plt.legend()
plt.show()


"""
This script performs a comprehensive machine learning workflow on the Iris dataset.
It includes data loading, preprocessing, model training, evaluation, and visualization.

Steps:
1. Load the dataset.
2. Perform initial data exploration and handle missing values.
3. Encode categorical variables.
4. Split the data into training and testing sets.
5. Scale the features.
6. Train and evaluate a Random Forest model.
7. Train and evaluate a Decision Tree model.
8. Visualize model performance using confusion matrices, feature importance, and decision trees.
9. Generate precision-recall curves for binary classification.
10. Perform hyperparameter tuning using GridSearchCV.
11. Conduct cross-validation with Stratified K-Fold.
12. Compare model performance.
13. Generate learning curves to diagnose overfitting/underfitting.
14. Log all steps for better traceability and debugging.
"""