from typing import Optional

from app.db.models import AnalysisResult
from app.db.session import SessionLocal


class AnalysisRepository:
    def save_analysis(
        self,
        result_label: str,
        result_confidence: Optional[float],
        recording_duration_seconds: float,
        processing_time_ms: int,
        model_ai: str,
    ) -> AnalysisResult:
        analysis = AnalysisResult(
            result_label=result_label,
            result_confidence=result_confidence,
            recording_duration_seconds=recording_duration_seconds,
            processing_time_ms=processing_time_ms,
            model_ai=model_ai,
        )
        with SessionLocal() as session:
            session.add(analysis)
            session.commit()
            session.refresh(analysis)
            return analysis
