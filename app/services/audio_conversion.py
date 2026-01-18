from fastapi import HTTPException, status
import ffmpeg


WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav"}
WAV_EXTENSIONS = {".wav"}


def convert_to_wav(audio_bytes: bytes, content_type: str, filename: str) -> bytes:
    if _is_wav(content_type, filename):
        return audio_bytes

    try:
        process = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="wav")
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        stdout, _ = process.communicate(input=audio_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ffmpeg is not installed.",
        ) from exc

    if process.returncode != 0 or not stdout:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode audio file.",
        )

    return stdout


def _is_wav(content_type: str, filename: str) -> bool:
    base_content_type = content_type.split(";", 1)[0].strip().lower()
    if base_content_type in WAV_CONTENT_TYPES:
        return True
    return filename.lower().endswith(tuple(WAV_EXTENSIONS))
