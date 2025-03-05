import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("web_incidents_for_regression.csv") 

df = df.replace({',': ''}, regex=True)



df.drop(columns=["short_description"], inplace=True)


categorical_cols = [
    "caller_id", "u_affected_user", "u_affected_user.location", "priority", "state", 
    "assignment_group", "assigned_to", "sys_updated_by", "u_choice_2", "category", 
    "incident_state", "contact_type", "business_service"
]

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

df = df.drop(columns=categorical_cols).join(encoded_features)
df = df.drop(columns=['incident_number'], errors='ignore') 


y = df["date_diff"]
df.drop(columns=["date_diff"], inplace=True)


results = {}
for col in df.columns:
    X = df[[col]]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[col] = {"R² Score": r2, "MAE": mae, "RMSE": rmse}


results_df = pd.DataFrame(results).T.sort_values(by="R² Score", ascending=False)


results_df.to_csv("linear_regression_results.csv")

print(results_df)
