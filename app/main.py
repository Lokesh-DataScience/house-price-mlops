from fastapi import FastAPI, HTTPException
import joblib
import os

from app.schema import HouseFeatures, PredictionResponse
from app.preprocess import preprocess_input
from app.logger import get_logger

logger = get_logger(__name__)

# Model versioning
MODEL_VERSION = "v1.0.0"
MODEL_PATH = os.getenv("MODEL_PATH", "app/model.pkl")

app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Production-ready ML API for predicting California house prices.",
    version=MODEL_VERSION,
)

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully")

@app.get("/")
def root():
    return {"message": "House Price Prediction API", "version": MODEL_VERSION}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        logger.info(f"Received prediction request: {features.model_dump()}")
        input_array = preprocess_input(features.model_dump())
        prediction = model.predict(input_array)[0]
        logger.info(f"Prediction: {prediction:.4f}")
        return PredictionResponse(
            predicted_price=round(float(prediction), 4),
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))