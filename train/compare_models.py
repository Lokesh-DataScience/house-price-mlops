"""
🏆 Train 6 different models and let MLflow help us pick the winner!
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("House_Price_Prediction")

# Load data
print("📥 Loading data...")
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define all experiments
experiments = [
    {
        "name": "linear_regression",
        "model": LinearRegression(),
        "params": {"model": "LinearRegression"},
        "flavor": "sklearn",
    },
    {
        "name": "ridge_alpha_1",
        "model": Ridge(alpha=1.0),
        "params": {"model": "Ridge", "alpha": 1.0},
        "flavor": "sklearn",
    },
    {
        "name": "ridge_alpha_10",
        "model": Ridge(alpha=10.0),
        "params": {"model": "Ridge", "alpha": 10.0},
        "flavor": "sklearn",
    },
    {
        "name": "random_forest_100",
        "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "params": {"model": "RandomForest", "n_estimators": 100, "max_depth": 10},
        "flavor": "sklearn",
    },
    {
        "name": "random_forest_300",
        "model": RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42),
        "params": {"model": "RandomForest", "n_estimators": 300, "max_depth": 20},
        "flavor": "sklearn",
    },
    {
        "name": "gradient_boosting",
        "model": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "params": {"model": "GradientBoosting", "n_estimators": 200, "max_depth": 5, "lr": 0.1},
        "flavor": "sklearn",
    },
    {
        "name": "xgboost_default",
        "model": XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42),
        "params": {"model": "XGBoost", "n_estimators": 300, "max_depth": 8, "lr": 0.1},
        "flavor": "xgboost",
    },
    {
    "name": "lightgbm_optimized",
    "model": LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    ),
        "params": {
            "model": "LightGBM",
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "flavor": "lightgbm",
    },
]

# Track best
results = []

print(f"\n🏁 Training {len(experiments)} models...\n" + "="*60)

for i, exp in enumerate(experiments, 1):
    print(f"\n[{i}/{len(experiments)}] 🚀 Training: {exp['name']}")
    
    with mlflow.start_run(run_name=exp["name"]) as run:
        # Log params
        mlflow.log_params(exp["params"])
        mlflow.set_tag("experiment_batch", "model_comparison_v1")
        
        # Train
        model = exp["model"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_metrics({"r2": r2, "rmse": rmse, "mae": mae})
        
        # Log model with appropriate flavor
        signature = infer_signature(X_train, model.predict(X_train))
        
        if exp["flavor"] == "xgboost":
            mlflow.xgboost.log_model(
                model, "model",
                signature=signature,
                registered_model_name="HousePricePredictor",
            )
        else:
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name="HousePricePredictor",
            )
        
        print(f"   ✅ R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")
        
        results.append({
            "name": exp["name"],
            "r2": r2,
            "rmse": rmse,
            "run_id": run.info.run_id,
        })

# Print summary
print("\n" + "="*60)
print("📊 FINAL RESULTS (sorted by R²):\n")
results.sort(key=lambda x: x["r2"], reverse=True)
for i, r in enumerate(results, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{emoji} {i}. {r['name']:25s} R²={r['r2']:.4f}  RMSE={r['rmse']:.4f}")

print(f"\n🏆 BEST MODEL: {results[0]['name']} (R²={results[0]['r2']:.4f})")
print(f"\n🌐 Compare all at: http://localhost:5000")
print("   Tip: Select multiple runs and click 'Compare'!")