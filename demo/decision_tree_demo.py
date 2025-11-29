# ==========================================
# Decision Tree Demo (Using From-Scratch Implementation)
# ==========================================

import pandas as pd
from algo.decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------
# Load Dataset
# ---------------------------------------
data = pd.read_csv("data/iris.csv")

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ---------------------------------------
# Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# Train model
# ---------------------------------------
model = DecisionTree(max_depth=5)
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

print("\nActual:", list(y_test[:10]))
print("Predicted:", list(pred[:10]))
