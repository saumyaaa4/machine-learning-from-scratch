# ================================
# Linear Regression
# ================================

# Introduction:
# Predicts a continuous value based on input features.
# Example: predicting house prices from size and location.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
X = data[['Size', 'Bedrooms']]  # Example features
y = data['Price']               # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Train Model
# ------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Make Predictions
# ------------------------
y_pred = model.predict(X_test)

# ------------------------
# Evaluate Model
# ------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# ------------------------
# Visualization
# ------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Understands relationships between variables.
# - Simple but powerful for regression tasks.
# - Can be used in real-world applications like predicting house prices, sales, etc.
