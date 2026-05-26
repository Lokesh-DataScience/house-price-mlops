from pydantic import BaseModel, Field

class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    currency: str = "USD (100k)"