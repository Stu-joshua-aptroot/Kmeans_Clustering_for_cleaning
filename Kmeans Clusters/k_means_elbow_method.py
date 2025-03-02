import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the encoded dataset
df = pd.read_csv("Web_incidents_for_clustering_encoded_all.csv")
df_cleaned = df.fillna(0)  # Replace NaNs with 0


# Normalize data (standardize all columns)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)


# 🔹 Step 2: Find the optimal number of clusters using the Elbow Method
wcss = []  # List to store Within-Cluster Sum of Squares (WCSS)
K_range = range(1, 11)  # Testing K from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)  # Save the WCSS score

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method to Determine Optimal k")
plt.show()

# 🔹 Step 3: Choose k and Perform K-Means Clustering
optimal_k = int(input("Enter the optimal number of clusters from the graph: "))  # Choose k based on the elbow point
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)  # Assign each row a cluster

# 🔹 Step 4: Analyze Cluster Results
print("\nCluster Counts:\n", df["Cluster"].value_counts())  # How many items per cluster
print("\nCluster Centers (Scaled Features):\n", kmeans.cluster_centers_)

# Save clustered data
df.to_csv("Web_incidents_with_clusters.csv", index=False)
print("Clustering complete! File saved as 'Web_incidents_with_clusters.csv'.")
