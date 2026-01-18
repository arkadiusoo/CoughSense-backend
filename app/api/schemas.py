from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., example="cough_positive")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.82)


class AnalysisResponse(BaseModel):
    result: Prediction
    analyzed_at: datetime = Field(..., example="2024-01-01T12:00:00Z")
