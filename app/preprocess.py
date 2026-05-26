import numpy as np

def preprocess_input(features: dict) -> np.ndarray:
    """Convert input dict to model-ready numpy array."""
    feature_order = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    return np.array([[features[f] for f in feature_order]])