from typing import Iterable


def validate_format(
    content_type: str,
    filename: str,
    allowed_content_types: Iterable[str],
    allowed_extensions: Iterable[str],
) -> None:
    base_content_type = content_type.split(";", 1)[0].strip().lower()
    if base_content_type and base_content_type not in allowed_content_types:
        raise ValueError("Unsupported audio format.")

    if filename and not _has_allowed_extension(filename, allowed_extensions):
        raise ValueError("Unsupported file extension.")


def validate_duration(duration_seconds: float, min_seconds: float, max_seconds: float) -> None:
    if duration_seconds < min_seconds:
        raise ValueError("Audio recording is too short.")
    if duration_seconds > max_seconds:
        raise ValueError("Audio recording is too long.")


def _has_allowed_extension(filename: str, allowed_extensions: Iterable[str]) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in allowed_extensions)
