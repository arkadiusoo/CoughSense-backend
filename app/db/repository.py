from app.ml.model import Prediction


class AnalysisRepository:
    def save_analysis(
        self,
        duration_seconds: float,
        prediction: Prediction,
        feature_vector_size: int,
    ) -> None:
        _ = (duration_seconds, prediction, feature_vector_size)
        # Placeholder for persistence layer integration.
