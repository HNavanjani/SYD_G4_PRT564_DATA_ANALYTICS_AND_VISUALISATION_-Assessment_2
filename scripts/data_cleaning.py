import pandas as pd
import os

# Load Excel dataset
df = pd.read_excel("data/avian_dataset.xlsx")
df.columns = df.columns.str.strip()  # Clean up column names

# Rename 'year' to 'Month'
df.rename(columns={"year": "Month"}, inplace=True)

# Remove unnamed or duplicate columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
df = df.loc[:, ~df.columns.duplicated()]

# Create new outbreak risk score
df["Outbreak_Risk_Score"] = df["Wasting"] + df["Found dead"] + df["Respiratory"]

# Save cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv("data/avian_cleaned.csv", index=False)

print("Data cleaned and saved as 'avian_cleaned.csv'")
