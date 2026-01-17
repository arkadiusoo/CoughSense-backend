from fastapi import APIRouter, File, UploadFile

from app.api.schemas import AnalysisResponse
from app.services.audio_analysis import analyze_cough_audio


router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    return await analyze_cough_audio(file)
