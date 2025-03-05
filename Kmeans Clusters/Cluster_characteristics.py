import pandas as pd

df_clusters = pd.read_csv("Web_incidents_with_clusters.csv")

df_cluster_1 = df_clusters[df_clusters["cluster"] == 1]


common_values = df_cluster_1.mode().iloc[0]

common_values.to_csv("Cluster_1_common_values.csv")

