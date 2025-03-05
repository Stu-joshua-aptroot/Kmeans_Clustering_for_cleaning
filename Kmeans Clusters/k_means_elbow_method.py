import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Web_incidents_for_clustering_encoded_all.csv")
df_cleaned = df.fillna(0)  



scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)



wcss = []  
K_range = range(1, 11)  

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)  


plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method to Determine Optimal k")
plt.show()


optimal_k = int(input("Enter the optimal number of clusters from the graph: "))  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)  

print("\nCluster Counts:\n", df["Cluster"].value_counts())  
print("\nCluster Centers (Scaled Features):\n", kmeans.cluster_centers_)


df.to_csv("Web_incidents_with_clusters.csv", index=False)
print("Clustering complete! File saved as 'Web_incidents_with_clusters.csv'.")
