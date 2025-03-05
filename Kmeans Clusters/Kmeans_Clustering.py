import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df_encoded = pd.read_csv("Web_incidents_for_clustering_encoded_all.csv")


df_encoded.fillna(0, inplace=True)


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_encoded["cluster"] = kmeans.fit_predict(df_scaled)

df_encoded.to_csv("Web_incidents_clustered.csv", index=False)


print("Cluster Sizes:")
print(df_encoded["cluster"].value_counts())


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_encoded.columns[:-1])
print("\nCluster Centroids:")
print(centroids)

for i in range(10): 
    print(f"\nCluster {i}:")
    print(df_encoded[df_encoded["cluster"] == i].mean())


pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_encoded["cluster"], cmap="viridis")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters (PCA Projection)")
plt.colorbar(label="Cluster")
plt.show()

df_encoded.to_csv("Web_incidents_with_clusters.csv", index=False)
