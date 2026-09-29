"""
main.py
-------
FastAPI backend for the Credit Card Fraud Detection dashboard.

Endpoints:
    GET  /metrics            -> default-threshold (0.5) metrics + dataset summary
    POST /evaluate            -> metrics recomputed at a caller-supplied threshold
    POST /predict              -> fraud probability + label for one transaction

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict

from model import fraud_model, MODEL_LOAD_ERROR, FraudModelUnavailable

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Threshold-adjustable fraud detection using a cached Logistic Regression model.",
    version="1.0.0",
)

# Serve the HTML/CSS/JavaScript frontend from the same FastAPI server.
# This means the complete dashboard opens at http://127.0.0.1:8000/
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "frontend"))

if not os.path.isdir(FRONTEND_DIR):
    raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Enable CORS so the static HTML/JS frontend (served separately, e.g. via
# `python -m http.server` or opened as a file) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="Fraud classification threshold, 0.0-1.0")


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Feature dict, e.g. {'Time': 100, 'V1': -1.2, ..., 'Amount': 49.99}"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0)


def ensure_model_ready():
    if fraud_model is None or not fraud_model.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model or dataset is unavailable. "
                f"Reason: {MODEL_LOAD_ERROR or 'unknown'}. "
                "Run `python train_model.py` in the backend/ folder first."
            ),
        )


@app.get("/", include_in_schema=False)
def root():
    """Serve the dashboard at the FastAPI root URL."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/metrics")
def get_metrics():
    """Returns metrics at the default 0.5 threshold, used on initial page load."""
    ensure_model_ready()
    try:
        result = fraud_model.evaluate_at_threshold(0.5)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {exc}")


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """Recomputes accuracy/recall/precision/F1/confusion matrix at the given threshold."""
    ensure_model_ready()
    try:
        result = fraud_model.evaluate_at_threshold(req.threshold)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate threshold: {exc}")


@app.post("/predict")
def predict(req: PredictRequest):
    """Predicts fraud probability + label for a single transaction's features."""
    ensure_model_ready()
    try:
        result = fraud_model.predict_single(req.features, req.threshold)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FraudModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
