import pandas as pd

df_clusters = pd.read_csv("Web_incidents_with_clusters.csv")


df_clusters['date_diff'] = pd.to_numeric(df_clusters['date_diff'], errors='coerce')

cluster_avg_date_diff = df_clusters.groupby("cluster")["date_diff"].mean().reset_index()

cluster_avg_date_diff.to_csv("Cluster_Average_DateDiff.csv", index=False)

print("New dataset saved as 'Cluster_Average_DateDiff.csv'.")

