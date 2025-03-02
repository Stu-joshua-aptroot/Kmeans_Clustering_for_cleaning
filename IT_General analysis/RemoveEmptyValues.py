import pandas as pd


df = pd.read_csv("Web_digital_no_duplicates.csv", encoding="UTF-8")


df_cleaned = df.dropna(subset=['u_choice_2'])


df_cleaned.to_csv("Web_incidents_cleaned.csv", index=False)
