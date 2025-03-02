import pandas as pd

# Load clustered data
df_clusters = pd.read_csv("Web_incidents_with_clusters.csv")

# Filter for Cluster 1
df_cluster_1 = df_clusters[df_clusters["cluster"] == 1]

# Find the most common value for each column
common_values = df_cluster_1.mode().iloc[0]

# Display results
print(common_values)
common_values.to_csv("Cluster_1_common_values.csv")

