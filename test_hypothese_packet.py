import pandas as pd
from scipy.stats import chi2_contingency

# Load your training data
df = pd.read_csv(r'C:\Users\hp\Saclay-ai\ML\train_data.csv')

# Create a contingency table (frequency table)
contingency_table = pd.crosstab(df['packet_duration'], df['target'])

# Apply the Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print results
print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p)

# Interpret, alpha = 0.05
if p < 0.05:
    print("❗ Reject the null hypothesis —i.e are dependent.")
else:
    print("✅ Cannot reject the null hypothesis —i.e appear independent.")