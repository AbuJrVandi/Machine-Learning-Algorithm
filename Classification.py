import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import logging

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logging.info("Starting the script...")

# Load the dataset
file_path = '/Users/user/Desktop/Dr. A. J/Machine_Learning/Classification Data.csv'
df = pd.read_csv(file_path)

# Display basic info about the dataset
logging.info("Displaying dataset info...")
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
logging.info("Checking for missing values...")
print("\nMissing Values:")
print(df.isnull().sum())

# Assume the last column is the target variable
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Convert categorical variables if needed
logging.info("Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
logging.info("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
logging.info("Scaling the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning using GridSearchCV
logging.info("Performing hyperparameter tuning...")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

# Train Logistic Regression model with best parameters
logging.info("Training the Logistic Regression model...")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
logging.info("Making predictions...")
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
logging.info("Plotting the confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC
logging.info("Plotting the ROC curve...")
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc:.2f}')

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
logging.info("Plotting the Precision-Recall curve...")
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
average_precision = average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Feature Importance
logging.info("Plotting feature importance...")
if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(best_model.coef_[0])})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=True)
    feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
    plt.title('Feature Importance')
    plt.show()

# Check class distribution
if len(np.unique(y_train)) < 2:
    logging.error("Training data must contain at least two classes.")
    raise ValueError("Training data must contain at least two classes.")

# Cross-Validation
logging.info("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5)  # Use StratifiedKFold
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# Learning Curve
logging.info("Plotting the learning curve...")
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.show()

# Model Persistence
logging.info("Saving the trained model...")
joblib.dump(best_model, 'logistic_regression_model.pkl')
logging.info("Model saved as 'logistic_regression_model.pkl'.")

logging.info("Script execution completed.")


"""
This script performs a logistic regression on a given dataset.
It includes data preprocessing, model training, evaluation, and visualization.

Steps:
1. Load the dataset.
2. Preprocess the data (handle missing values, encode categorical variables).
3. Split the data into training and testing sets.
4. Scale the features.
5. Train a logistic regression model.
6. Evaluate the model using accuracy, classification report, confusion matrix, and ROC curve.
7. Visualize the results.
8. Additional enhancements include feature importance, cross-validation, hyperparameter tuning, precision-recall curve, learning curve, model persistence, and logging.
"""