import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1️⃣ Load the Encoded CSV
df_encoded = pd.read_csv("Web_incidents_for_clustering_encoded_all.csv")

# 2️⃣ Handle Missing Values (Replace NaNs with 0)
df_encoded.fillna(0, inplace=True)

# 3️⃣ Scale the Data (Important for K-Means)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# 4️⃣ Apply K-Means Clustering (K = 10)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_encoded["cluster"] = kmeans.fit_predict(df_scaled)

# 5️⃣ Save Clustered Data
df_encoded.to_csv("Web_incidents_clustered.csv", index=False)

# 6️⃣ Analyze Clusters
print("Cluster Sizes:")
print(df_encoded["cluster"].value_counts())

# Cluster Centroids (to see what defines each cluster)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_encoded.columns[:-1])
print("\nCluster Centroids:")
print(centroids)

# Print Average Feature Values per Cluster
for i in range(10): 
    print(f"\nCluster {i}:")
    print(df_encoded[df_encoded["cluster"] == i].mean())

# 7️⃣ Visualize Clusters with PCA (2D Projection)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_encoded["cluster"], cmap="viridis")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters (PCA Projection)")
plt.colorbar(label="Cluster")
plt.show()

df_encoded.to_csv("Web_incidents_with_clusters.csv", index=False)
