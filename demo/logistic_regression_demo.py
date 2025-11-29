# ==========================================
# Logistic Regression Demo (Using From-Scratch Implementation)
# ==========================================

import pandas as pd
from algo.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------
# Load Dataset
# ---------------------------------------
data = pd.read_csv("data/iris.csv")

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Convert labels to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ---------------------------------------
# Train-test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# Train Model
# ---------------------------------------
model = LogisticRegression(learning_rate=0.01, epochs=2000)
model.fit(X_train.values, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------
pred = model.predict(X_test.values)

# ---------------------------------------
# Accuracy
# ---------------------------------------
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# ---------------------------------------
# Sample predictions
# ---------------------------------------
print("\nActual:", list(y_test[:10]))
print("Predicted:", list(pred[:10]))
