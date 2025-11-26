# ================================
# K-Means Clustering
# ================================

# Introduction:
# Groups data points into clusters based on similarity.
# Example: customer segmentation for marketing.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------
# Load Dataset
# ------------------------
# Replace 'dataset.csv' with your dataset file
data = pd.read_csv('dataset.csv')
print("First 5 rows of dataset:")
print(data.head())

# ------------------------
# Data Preparation
# ------------------------
# Select features (X) for clustering
X = data[['Feature1', 'Feature2']]  # Replace with actual features

# ------------------------
# Apply K-Means
# ------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add cluster labels to dataset
data['Cluster'] = labels
print("\nData with cluster labels:")
print(data.head())

# ------------------------
# Visualization
# ------------------------
plt.scatter(X['Feature1'], X['Feature2'], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], s=200, color='red', marker='X')
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("K-Means Clustering")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Groups similar data points together.
# - Useful for market segmentation, anomaly detection, and pattern recognition.
# - Number of clusters (k) is a key parameter to tune.
