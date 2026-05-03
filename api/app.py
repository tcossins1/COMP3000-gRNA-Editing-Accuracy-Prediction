from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import MODEL_PATH, MODEL_NAME
from src.service.predictor import EfficiencyPredictor

# --- Request/Response schemas ---
class PredictRequest(BaseModel):
    sequence: str

class PredictResponse(BaseModel):
    sequence: str
    prediction: float
    features: dict
    model: str


app = FastAPI(title="ForeCas9 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Must match training feature order
FEATURE_COLUMNS = ["gc_content"]

# Load model once on startup
predictor = EfficiencyPredictor(MODEL_PATH, feature_columns=FEATURE_COLUMNS)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predictor.predict(req.sequence)
        return {
            "sequence": result.sequence,
            "prediction": result.prediction,
            "features": result.features,
            "model": MODEL_NAME,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")