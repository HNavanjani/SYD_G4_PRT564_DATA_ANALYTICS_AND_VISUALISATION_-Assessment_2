import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

df = pd.read_csv("data/avian_cleaned.csv")

# Drop rows with missing values (fixes model training error)
df = df.dropna()

# Define features and target
X = df.drop(columns=["Month", "Outbreak_Risk_Score"])
y = df["Outbreak_Risk_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save all outputs
os.makedirs("outputs/results", exist_ok=True)
joblib.dump(X_train_scaled, "outputs/results/X_train.pkl")
joblib.dump(X_test_scaled, "outputs/results/X_test.pkl")
joblib.dump(y_train, "outputs/results/y_train.pkl")
joblib.dump(y_test, "outputs/results/y_test.pkl")
joblib.dump(scaler, "outputs/results/scaler.pkl")

print("Preprocessing complete.")
