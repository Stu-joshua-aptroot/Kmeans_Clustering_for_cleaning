import pandas as pd


df = pd.read_csv("Web & digital - Tickets.csv", encoding="Windows-1252")  # Use cleaned file

df_no_duplicates = df.sort_values(by="opened_at", ascending=False).drop_duplicates(subset=['number'], keep='first')

df_no_duplicates.to_csv("Web_digital_no_duplicates.csv", index=False)

