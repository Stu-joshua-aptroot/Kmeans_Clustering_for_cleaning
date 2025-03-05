import pandas as pd
import scipy.stats as stats


df = pd.read_csv("it_tickets_no_missing.csv")


contingency_table = pd.crosstab(df["u_choice_2"], df["business_service"])


chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)


summary_df = pd.DataFrame({
    "Chi-square Statistic": [chi2],
    "Degrees of Freedom": [dof],
    "P-value": [p]
})


contingency_table.to_csv("contingency_table.csv")
expected_df.to_csv("expected_frequencies.csv")
summary_df.to_csv("chi_square_summary.csv", index=False)

print("CSV files generated:")
print(" - contingency_table.csv")
print(" - expected_frequencies.csv")
print(" - chi_square_summary.csv")

alpha = 0.05  
if p < alpha:
    print(" Significant Relationship Found: Departments are likely linked to specific IT issues.")
else:
    print(" No Significant Relationship: Issue types are evenly distributed across departments.")
