from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., example="cough_positive")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.82)


class AnalysisResponse(BaseModel):
    prediction: Prediction
    duration_seconds: float = Field(..., gt=0.0, example=2.35)
    feature_vector_size: int = Field(..., gt=0, example=58)
