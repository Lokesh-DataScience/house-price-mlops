"""
🚀 Find the best model and promote it to Production stage.
"""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "HousePricePredictor"

print(f"🔍 Searching all versions of '{MODEL_NAME}'...\n")

# Get all model versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

best_version = None
best_r2 = -float("inf")

print(f"{'Version':<10} {'R²':<10} {'Stage':<15} {'Run ID':<20}")
print("-" * 60)

for v in versions:
    run = client.get_run(v.run_id)
    r2 = run.data.metrics.get("r2", -float("inf"))
    stage = v.current_stage
    print(f"{v.version:<10} {r2:<10.4f} {stage:<15} {v.run_id[:18]}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_version = v.version

print(f"\n🏆 BEST MODEL FOUND!")
print(f"   Version: {best_version}")
print(f"   R² Score: {best_r2:.4f}")

# Transition the best model to Production
print(f"\n🚀 Promoting version {best_version} to PRODUCTION...")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=best_version,
    stage="Production",
    archive_existing_versions=True,
)

# Add a description
client.update_model_version(
    name=MODEL_NAME,
    version=best_version,
    description=f"🏆 Promoted to production. R²={best_r2:.4f}",
)

print(f"✅ Version {best_version} is now in PRODUCTION!")
print(f"\n🌐 View at: http://localhost:5000/#/models/{MODEL_NAME}")