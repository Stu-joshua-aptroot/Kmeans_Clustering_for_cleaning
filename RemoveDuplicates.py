import pandas as pd

# Load the dataset
df = pd.read_csv("it_tickets_no_missing.csv", encoding="utf-8")  # Use cleaned file

# Remove duplicates, keeping the latest entry
df_no_duplicates = df.sort_values(by="opened_at", ascending=False).drop_duplicates(subset=['number'], keep='first')

# Save the updated dataset
df_no_duplicates.to_csv("it_tickets_no_duplicates.csv", index=False)

