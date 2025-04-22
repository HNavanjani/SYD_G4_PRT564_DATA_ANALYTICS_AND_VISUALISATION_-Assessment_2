import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

X_train = joblib.load("outputs/results/X_train.pkl")
y_train = joblib.load("outputs/results/y_train.pkl")

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Save models
joblib.dump(lr, "outputs/results/linear_model.pkl")
joblib.dump(rf, "outputs/results/rf_model.pkl")

print("✅ Models trained and saved.")
