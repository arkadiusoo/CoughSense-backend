from fastapi import APIRouter, File, UploadFile

from app.api.schemas import AnalysisResponse
from app.services.audio_analysis import analyze_cough_audio


router = APIRouter(prefix="/api", tags=["analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze cough audio",
    description=(
        "Accepts a single audio file and returns the predicted label with confidence. "
        "Supported inputs should be short recordings containing cough, breath, or other "
        "ambient sound. The response includes the model decision and a confidence score."
    ),
)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    return await analyze_cough_audio(file)
