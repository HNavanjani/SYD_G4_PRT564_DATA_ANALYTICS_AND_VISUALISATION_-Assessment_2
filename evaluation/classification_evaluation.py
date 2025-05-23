import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load models and encoders
log_model = joblib.load("models/classification/logistic_model.pkl")
rf_model = joblib.load("models/classification/random_forest_model.pkl")
scaler = joblib.load("models/classification/scaler.pkl")
label_encoder = joblib.load("models/classification/label_encoder.pkl")

# Load data
df = pd.read_csv("data/avian_cleaned.csv")

# Reconstruct labels
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
X = df.drop(columns=["Month", "Outbreak_Risk_Score", "Risk_Level"])
y = label_encoder.transform(df["Risk_Level"])
X_scaled = scaler.transform(X)

# Predictions
log_preds = log_model.predict(X_scaled)
rf_preds = rf_model.predict(X_scaled)

# Evaluation
log_report = classification_report(y, log_preds, target_names=label_encoder.classes_)
rf_report = classification_report(y, rf_preds, target_names=label_encoder.classes_)

print("Logistic Regression Report:\n", log_report)
print("Random Forest Report:\n", rf_report)

# Confusion matrix for RF
conf_matrix = confusion_matrix(y, rf_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
