import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model R² Score: {model.score(X_test, y_test):.4f}")

# Save model
joblib.dump(model, "app/model.pkl")
print("✅ Model saved to app/model.pkl")