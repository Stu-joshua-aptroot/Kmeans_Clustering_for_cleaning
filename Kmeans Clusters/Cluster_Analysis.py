import pandas as pd


df_clusters = pd.read_csv("Web_incidents_with_clusters.csv")


print(df_clusters.groupby("cluster").mean())  


print(df_clusters["cluster"].value_counts())
df_clusters.groupby("cluster")["date_diff"].mean().sort_values(ascending=False)

