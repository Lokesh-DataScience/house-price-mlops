"""
🔬 First MLflow-Tracked Training Run
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# ============================================
# 1️⃣ Configure MLflow
# ============================================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("House_Price_Prediction")

print("🔬 MLflow tracking URI:", mlflow.get_tracking_uri())

# ============================================
# 2️⃣ Load Data
# ============================================
print("\n📥 Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================
# 3️⃣ Define Hyperparameters
# ============================================
params = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": 42,
}

# ============================================
# 4️⃣ Start MLflow Run
# ============================================
with mlflow.start_run(run_name="rf_baseline_v1") as run:
    
    print(f"\n🚀 Starting MLflow run: {run.info.run_id}")
    
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("dataset", "california_housing")
    
    # Train the model
    print("🧠 Training RandomForest...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log feature importances
    for feature, importance in zip(X.columns, model.feature_importances_):
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # Add tags
    mlflow.set_tag("developer", "your_name")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("stage", "experimentation")
    
    # Log the model with signature & input example
    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name="HousePricePredictor",
    )
    
    # ============================================
    # 5️⃣ Print Results
    # ============================================
    print(f"\n✅ Training Complete!")
    print(f"   📊 R² Score: {r2:.4f}")
    print(f"   📉 RMSE:     {rmse:.4f}")
    print(f"   📐 MAE:      {mae:.4f}")
    print(f"\n🌐 View this run at:")
    print(f"   http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")