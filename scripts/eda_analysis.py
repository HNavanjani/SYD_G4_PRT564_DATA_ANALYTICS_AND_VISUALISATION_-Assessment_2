import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/avian_cleaned.csv")

# Convert Month to string to fix x-axis plotting issue
df["Month"] = df["Month"].astype(str)

os.makedirs("outputs/figures", exist_ok=True)

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_heatmap.png")
plt.close()

# Trend line of key symptoms
symptoms = ["Wasting", "Found dead", "Respiratory"]
for col in symptoms:
    plt.plot(df["Month"], df[col], label=col)

plt.title("Monthly Symptom Trends")
plt.xlabel("Month")
plt.ylabel("Case Count")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/symptom_trends.png")
plt.close()

print("EDA plots saved.")
