# CoughSense Backend

This repository contains the backend part of the CoughSense system – a web-based AI application for analyzing cough audio signals.

The backend is responsible for:
- exposing a REST API,
- receiving and validating audio recordings,
- extracting audio features,
- running inference using a trained AI model,
- storing anonymized results in a PostgreSQL database.

### Technology stack
- Python
- FastAPI
- PyTorch
- PostgreSQL
- Docker

### Architectural assumptions
The backend is designed as a modular monolith with a clear separation of concerns:
- API layer (FastAPI),
- application and business logic,
- machine learning layer (feature extraction and inference),
- persistence layer.

The system does not store raw audio files or any personally identifiable user data.
