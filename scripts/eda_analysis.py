import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/avian_cleaned.csv")

# Ensure Month is string for readable plots
df["Month"] = df["Month"].astype(str)

# Create output directory
output_dir = "outputs/eda_enhanced"
os.makedirs(output_dir, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# 2. Symptom Trends Over Time
symptoms = ["Wasting", "Found dead", "Respiratory"]
plt.figure(figsize=(14, 6))
for col in symptoms:
    plt.plot(df["Month"], df[col], label=col, marker='o')
plt.title("Monthly Symptom Trends")
plt.xlabel("Month")
plt.ylabel("Symptom Count")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/symptom_trends.png")
plt.close()

# 3. Histogram of Outbreak Risk Scores
plt.figure(figsize=(8, 5))
sns.histplot(df["Outbreak_Risk_Score"], bins=15, kde=True)
plt.title("Distribution of Outbreak Risk Score")
plt.xlabel("Outbreak Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_dir}/risk_score_distribution.png")
plt.close()

# 4. Boxplot of Core Symptoms
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[symptoms])
plt.title("Boxplot of Core Symptoms")
plt.ylabel("Reported Cases")
plt.tight_layout()
plt.savefig(f"{output_dir}/symptom_boxplots.png")
plt.close()

# 5. Pairplot of Symptoms
sns.pairplot(df[symptoms])
plt.suptitle("Symptom Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/symptom_pairplot.png")
plt.close()

print(f"Enhanced EDA plots saved to: {output_dir}")
