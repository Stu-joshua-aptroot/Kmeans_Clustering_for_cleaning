import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("linear_regression_results.csv")


categories = df["Unnamed: 0"]
r2_scores = df["R² Score"]
mae = df["MAE"]
rmse = df["RMSE"]

plt.figure(figsize=(14, 12))

plt.subplot(3, 1, 1)
plt.bar(categories, r2_scores, color='skyblue')
plt.title("R² Score per Category")
plt.xlabel("Category")
plt.ylabel("R² Score")
plt.xticks(rotation=90)

plt.subplot(3, 1, 2)
plt.bar(categories, mae, color='orange')
plt.title("Mean Absolute Error (MAE) per Category")
plt.xlabel("Category")
plt.ylabel("MAE (days)")
plt.xticks(rotation=90)

plt.subplot(3, 1, 3)
plt.bar(categories, rmse, color='green')
plt.title("Root Mean Squared Error (RMSE) per Category")
plt.xlabel("Category")
plt.ylabel("RMSE (days)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
