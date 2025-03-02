import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the CSV file with one-hot encoded data
df = pd.read_csv("Web_incidents_for_regression_encoded.csv")

# Ensure the target column is present
if 'date_diff' not in df.columns:
    raise ValueError("The target column 'date_diff' is missing.")

# Separate features and target
X = df.drop(columns=["date_diff"])
y = df["date_diff"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Random Forest Regression ###
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluate Random Forest performance
rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))


print("Random Forest Results:")
print("------------------------")
print(f"R² Score: {rf_r2:.3f}")
print(f"Mean Absolute Error (MAE): {rf_mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.3f}\n")

# Feature Importances from Random Forest
importances_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importances:")
print(importances_rf, "\n")

### Gradient Boosting Regression ###
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# Evaluate Gradient Boosting performance
gb_r2 = r2_score(y_test, gb_preds)
gb_mae = mean_absolute_error(y_test, gb_preds)


print("Gradient Boosting Results:")
print("----------------------------")
print(f"R² Score: {gb_r2:.3f}")
print(f"Mean Absolute Error (MAE): {gb_mae:.3f}")

importances_rf.to_csv("Random_forest_Reults.csv")
