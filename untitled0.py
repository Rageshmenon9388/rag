# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 23:52:14 2025

@author: rages
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 1: Load the dataset
df = pd.read_csv("simulated_health_wellness_data.csv")  # <-- Replace with your actual path

# Step 2: Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 3: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title("Explained Variance by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# Reduce to 2 dimensions for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Step 4: KMeans Clustering - Original Data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels_orig = kmeans.fit_predict(X_scaled)
silhouette_orig = silhouette_score(X_scaled, kmeans_labels_orig)
db_orig = davies_bouldin_score(X_scaled, kmeans_labels_orig)

# Step 5: KMeans Clustering - PCA Data
kmeans_pca = KMeans(n_clusters=3, random_state=42)
kmeans_labels_pca = kmeans_pca.fit_predict(X_pca_2d)
silhouette_pca = silhouette_score(X_pca_2d, kmeans_labels_pca)
db_pca = davies_bouldin_score(X_pca_2d, kmeans_labels_pca)

# Step 6: Visualize Clusters in PCA Space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=kmeans_labels_pca, palette='Set2')
plt.title("K-Means Clusters (PCA-Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# Step 7: Print Evaluation Metrics
print("CLUSTERING EVALUATION")
print("----------------------")
print(f"K-Means on Original Data - Silhouette Score: {silhouette_orig:.4f}")
print(f"K-Means on Original Data - Davies-Bouldin Score: {db_orig:.4f}")
print()
print(f"K-Means on PCA Data - Silhouette Score: {silhouette_pca:.4f}")
print(f"K-Means on PCA Data - Davies-Bouldin Score: {db_pca:.4f}")
