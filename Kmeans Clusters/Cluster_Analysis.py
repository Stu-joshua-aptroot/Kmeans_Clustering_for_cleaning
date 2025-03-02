import pandas as pd

# Load clustered data
df_clusters = pd.read_csv("Web_incidents_with_clusters.csv")

# View basic cluster statistics
print(df_clusters.groupby("cluster").mean())  # For numeric columns

# Check cluster sizes
print(df_clusters["cluster"].value_counts())
df_clusters.groupby("cluster")["date_diff"].mean().sort_values(ascending=False)

