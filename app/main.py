"""
🏠 House Price API (v2.0) — Now powered by MLflow Model Registry!
"""

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os

from app.schema import HouseFeatures, PredictionResponse
from app.preprocess import preprocess_input
from app.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Configuration
# ============================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "HousePricePredictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="MLflow-powered Production ML API",
    version="2.0.0",
)

# Global state
model = None
model_version = None
model_run_id = None

# ============================================
# Load model from MLflow Registry
# ============================================
def load_production_model():
    global model, model_version, model_run_id
    
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"📥 Loading from MLflow Registry: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Get version info
    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if latest:
        model_version = latest[0].version
        model_run_id = latest[0].run_id
    
    logger.info(f"✅ Loaded {MODEL_NAME} v{model_version} ({MODEL_STAGE})")

# ============================================
# Startup event
# ============================================
@app.on_event("startup")
def startup():
    try:
        load_production_model()
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

# ============================================
# Routes
# ============================================
@app.get("/")
def root():
    return {
        "message": "🏠 House Price Prediction API",
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_stage": MODEL_STAGE,
        "mlflow_ui": MLFLOW_TRACKING_URI,
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_version": model_version,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Prediction request: {features.dict()}")
        input_array = preprocess_input(features.dict())
        prediction = model.predict(input_array)[0]
        logger.info(f"Predicted: {prediction:.4f}")
        return PredictionResponse(
            predicted_price=round(float(prediction), 4),
            model_version=f"v{model_version}",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload_model():
    """🔄 Hot-reload the latest production model (zero downtime!)"""
    try:
        load_production_model()
        return {
            "status": "✅ reloaded",
            "model_version": model_version,
            "model_stage": MODEL_STAGE,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    """ℹ️ Get detailed info about the currently loaded model."""
    if model_run_id is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    client = MlflowClient()
    run = client.get_run(model_run_id)
    
    return {
        "model_name": MODEL_NAME,
        "version": model_version,
        "stage": MODEL_STAGE,
        "run_id": model_run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }