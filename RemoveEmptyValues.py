import pandas as pd

# Load the dataset
df = pd.read_csv("it_tickets.csv", encoding="windows-1252")

# Remove rows where any column has missing values
df_cleaned = df.dropna(subset=['business_service'])

# Save the cleaned dataset
df_cleaned.to_csv("it_tickets_no_missing.csv", index=False)
