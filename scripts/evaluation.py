import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

X_test = joblib.load("outputs/results/X_test.pkl")
y_test = joblib.load("outputs/results/y_test.pkl")

lr = joblib.load("outputs/results/linear_model.pkl")
rf = joblib.load("outputs/results/rf_model.pkl")

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("R² Score: ", r2_score(y_true, y_pred))
    print("RMSE:     ", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:      ", mean_absolute_error(y_true, y_pred))

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)

# Save predicted vs actual to CSV
results_df = pd.DataFrame({
    "Month": range(1, len(y_test)+1),
    "Actual_Risk": y_test,
    "Predicted_RF": rf_pred,
    "Predicted_LR": lr_pred
})
results_df.to_csv("outputs/results/predicted_outbreak_risks.csv", index=False)
