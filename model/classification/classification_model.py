import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("data/avian_cleaned.csv")

# Create risk level labels
def classify_risk(score):
    if score <= 5:
        return "Low"
    elif score <= 10:
        return "Medium"
    elif score <= 15:
        return "High"
    else:
        return "Critical"

df["Risk_Level"] = df["Outbreak_Risk_Score"].apply(classify_risk)

# Features and labels
X = df.drop(columns=["Month", "Outbreak_Risk_Score", "Risk_Level"])
y = df["Risk_Level"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models and encoders
os.makedirs("models/classification", exist_ok=True)
joblib.dump(log_model, "models/classification/logistic_model.pkl")
joblib.dump(rf_model, "models/classification/random_forest_model.pkl")
joblib.dump(scaler, "models/classification/scaler.pkl")
joblib.dump(label_encoder, "models/classification/label_encoder.pkl")
