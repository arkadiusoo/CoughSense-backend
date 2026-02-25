# CoughSense Backend API

## Short Description
This project is a Python backend service for cough audio analysis.  
It exposes an HTTP API built with FastAPI.  
The service accepts one audio file, validates and converts it, extracts audio features, and runs AI inference.  
It saves each analysis result in PostgreSQL.  
At startup, it creates database tables with SQLAlchemy metadata.

## Technologies Used
Technologies below are taken from source code and configuration files.

- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- PostgreSQL
- psycopg2
- NumPy
- SciPy
- Librosa
- ffmpeg-python (requires `ffmpeg` binary in the system)
- PyTorch
- scikit-learn
- Pillow
- Matplotlib
- Docker Compose (database service)
- Pytest

## Architecture Overview
The backend is a modular monolith.

- API layer: `app/main.py`, `app/api/routes.py`, `app/api/schemas.py`
- Service layer: `app/services/audio_analysis.py`
- Audio processing: `app/audio/*`
- AI inference: `ml_2/model.py` for `CoughSenseModel_2-0` and `app/ml/model.py` for `CoughSenseModel_1-0` fallback
- Persistence layer: `app/db/*` + PostgreSQL init SQL in `db/init/001-init.sql`

Request flow:
1. Client sends a file to `POST /api/analyze`.
2. Backend validates format and duration.
3. Backend converts input to WAV if needed.
4. Backend extracts numeric audio features.
5. Backend runs model prediction.
6. Backend saves result metadata to PostgreSQL.
7. Backend returns JSON response.

## Folder Structure
Main backend folders:

```text
app/
  api/        # HTTP routes and response schemas
  audio/      # audio validation, conversion, loading, features, spectrograms
  db/         # SQLAlchemy base, models, session, repository
  ml/         # legacy model loader and 1D CNN training code
  services/   # application logic used by API routes
  main.py     # FastAPI app, CORS, startup table creation
db/
  init/       # SQL init scripts for PostgreSQL (pgcrypto extension)
ml_2/
  model.py    # runtime hybrid model (CoughSenseModel_2-0)
  results/    # expected model result/config files used at runtime
artifacts/
  coughsense_model_1_0/  # legacy model artifacts used as fallback
scripts/
  *.py        # training and analysis scripts for model pipelines
tests/
  *.py        # backend tests for audio conversion and feature extraction
```

## API Overview
Base router prefix: `/api`

### `POST /api/analyze`
- Input: `multipart/form-data` with one file field: `file`
- Allowed content types: `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp3`, `audio/webm`
- Allowed extensions: `.wav`, `.mp3`, `.webm`
- Duration limits: `0.1` to `60.0` seconds
- Note: response currently returns only `label` and `analyzed_at`. `result_confidence` is stored in the database.

Current response shape:

```json
{
  "result": {
    "label": "string"
  },
  "analyzed_at": "2024-01-01T12:00:00+00:00"
}
```

### `GET /api/health`
- Response: `{"status":"ok"}`

## Database Integration
- Database engine: PostgreSQL via SQLAlchemy.
- Connection source: `DATABASE_URL` environment variable in `app/db/session.py`.
- Default URL in code:  
  `postgresql+psycopg2://coughsense:coughsense@localhost:5432/coughsense`
- Table: `analysis_results` (`app/db/models.py`)
- Saved fields: `id`, `analyzed_at`, `result_label`, `result_confidence`, `recording_duration_seconds`, `processing_time_ms`, `model_ai`
- Startup behavior: `Base.metadata.create_all(bind=engine)` creates tables if missing.
- SQL init file enables `pgcrypto`: `db/init/001-init.sql`.

## AI Model Integration
- Main runtime path is in `app/services/audio_analysis.py`.
- The service first tries to load `CoughSenseModel_2-0` from `ml_2/results`.
- Model 2.0 structure: stage 1 classifies signal type (`cough` or `breath`), then stage 2 runs the related submodel. Each submodel fuses CNN and KNN probabilities.
- If Model 2.0 cannot be loaded, backend falls back to `CoughSenseModel_1-0` (`app/ml/model.py`).
- If Model 1.0 artifacts are missing, backend falls back to a stub predictor.
- Audio preprocessing used for inference: convert input to WAV, load mono waveform, compute MFCC/chroma/spectral summaries, and generate mel spectrogram images for hybrid submodels.

## Environment Variables
Visible backend environment variables:

- `DATABASE_URL`: used by SQLAlchemy. It has a default value in code.

Variables defined in `docker-compose.yml` for database container:

- `POSTGRES_DB=coughsense`
- `POSTGRES_USER=coughsense`
- `POSTGRES_PASSWORD=coughsense`
