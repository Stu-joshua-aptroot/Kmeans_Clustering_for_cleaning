import pandas as pd


df = pd.read_csv("Web_incidents_for_regression.csv")


categorical_cols = ['caller_id', 'u_affected_user', 'u_affected_user.location',
    'priority', 'state', 'assignment_group', 'assigned_to', 'sys_updated_by',
    'u_choice_2', 'category', 'incident_state', 'contact_type', 'business_service'
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


df_encoded.to_csv("Web_incidents_for_clustering_encoded_all.csv", index=False)

print("One-hot encoding complete. File saved as 'Web_incidents_for_clustering_encoded_all.csv'.")
