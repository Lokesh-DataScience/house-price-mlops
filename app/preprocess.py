import pandas as pd

# The order MUST match the training data column order
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

def preprocess_input(features: dict) -> pd.DataFrame:
    """
    Convert input dict to a pandas DataFrame with named columns
    matching the MLflow model signature.
    """
    return pd.DataFrame([{col: features[col] for col in FEATURE_ORDER}])