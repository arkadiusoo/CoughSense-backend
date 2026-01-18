from sqlalchemy import Column, DateTime, Float, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    analyzed_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    result_label = Column(String, nullable=False)
    result_confidence = Column(Float, nullable=True)
    recording_duration_seconds = Column(Float, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    model_ai = Column(String, nullable=False)
