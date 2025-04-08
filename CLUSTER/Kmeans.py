# FOR CLUSTERING ALSO CONSISTS ELBOW AND SHILHOUTEE SCORE 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Step 1: Min-Max Scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:-1]), columns=df.columns[1:-1])

# Step 2: PCA for Dimensionality Reduction
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=["PC1", "PC2"])

# Step 3: Finding Optimal Clusters (Elbow Method)
wcss = []  # Within-cluster sum of squares
silhouette_scores = []  # Silhouette scores
K_range = range(2, 10)  # Testing clusters from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    wcss.append(kmeans.inertia_)  # Append WCSS value
    silhouette_scores.append(silhouette_score(df_pca, labels))  # Append Silhouette score

# Plot Elbow Method
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(K_range, wcss, marker='o', linestyle='-', color='b', label="WCSS (Elbow Method)")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method for Optimal K")
ax1.legend()
plt.grid()
plt.show()

# Plot Silhouette Score
fig, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='r', label="Silhouette Score")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score for Different K")
ax2.legend()
plt.grid()
plt.show()

# Step 4: K-Means Clustering with Optimal K
optimal_k = 4  # Choose based on elbow and silhouette score analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_pca["Cluster"] = kmeans.fit_predict(df_pca)

# Step 5: Sorting Clusters by Size
cluster_sizes = df_pca["Cluster"].value_counts().sort_values().index  # Get sorted cluster order
cluster_mapping = {old: new for new, old in enumerate(cluster_sizes)}  # Create mapping
df_pca["Cluster"] = df_pca["Cluster"].map(cluster_mapping)  # Apply mapping

# Step 6: Get Sorted Centroids
centroids = kmeans.cluster_centers_
centroids = np.array([centroids[old] for old in cluster_sizes])  # Reorder centroids

# Assign readable cluster labels
cluster_labels = {0: "Good", 1: "Best", 2: "Average", 3: "Poor"}
df_pca["Cluster"] = df_pca["Cluster"].replace(cluster_labels)

# Ensure legend order follows "Best" → "Good" → "Average" → "Poor"
cluster_order = ["Best", "Good", "Average", "Poor"]

# Step 7: Scatter Plot with Sorted Clusters
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor("#e2ffce")

sns.scatterplot(
    data=df_pca, x="PC1", y="PC2", hue="Cluster",
    palette=palette, s=100, alpha=0.8, edgecolor="black",
    ax=ax, zorder=2, hue_order=cluster_order  # Ensuring correct legend order
)

# Plot Centroids
plt.scatter(
    centroids[:, 0], centroids[:, 1], marker="X",
    s=200, color="black", edgecolor="white",
    label="Centroids", zorder=3
)

# Titles and legend
ax.set_title("Customer Clusters with Centroids")
ax.legend(title="Cluster")
ax.grid(True, axis="y", linestyle="--", zorder=1)
plt.show()



# TO GET THE SAME CLUSTER COLUMN IN MAIN DF >>>just dont change the sequence

df["Cluster"] = df_pca["Cluster"]