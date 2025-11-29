# ================================
# Logistic Regression
# ================================

# Introduction:
# Predicts a binary outcome (yes/no, 0/1).
# Example: predicting whether a customer will buy a product.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Load Dataset
# ------------------------
# Replace 'dataset.csv' with your dataset file in the repo
data = pd.read_csv('dataset.csv')
print("First 5 rows of dataset:")
print(data.head())

# ------------------------
# Data Preparation
# ------------------------
# Select features (X) and target (y)
X = data[['Feature1', 'Feature2']]  # Replace with actual feature columns
y = data['Target']                  # Replace with target column

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Train Model
# ------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------
# Make Predictions
# ------------------------
y_pred = model.predict(X_test)

# ------------------------
# Evaluate Model
# ------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Good for classification problems.
# - Outputs probabilities of classes.
# - Widely used in marketing, finance, and medical predictions.
