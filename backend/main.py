"""
FraudShield AI — FastAPI Backend
Endpoints: GET /metrics, POST /predict, GET /static/{filename}
"""

import os, json
from pathlib import Path
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "model.pkl"
META_PATH   = BASE_DIR / "metrics.json"
FEAT_PATH   = BASE_DIR / "feature_names.json"
STATIC_DIR  = BASE_DIR.parent / "eda_xgboost_results"

# ── Load model on startup ──────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found — run train_and_save.py first")

model    = joblib.load(MODEL_PATH)
metrics  = json.loads(META_PATH.read_text())
features = json.loads(FEAT_PATH.read_text())

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="FraudShield AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve EDA images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response Schemas ─────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    step: float = 1
    type: str   = "TRANSFER"      # PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
    amount: float = 0.0
    oldbalanceOrg: float  = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    isFlaggedFraud: int = 0       # kept for input parity; not used as feature


class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    features_used: dict


# ── Feature engineering (mirrors aiml_eda_xgboost.py) ─────────────────────────
TYPE_MAP = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}

def engineer_features(req: TransactionRequest) -> np.ndarray:
    type_enc          = TYPE_MAP.get(req.type.upper(), -1)
    orig_diff         = req.oldbalanceOrg  - req.newbalanceOrig
    dest_diff         = req.newbalanceDest - req.oldbalanceDest
    orig_balance_ratio= req.newbalanceOrig  / (req.oldbalanceOrg  + 1) if req.oldbalanceOrg  > 0 else 0
    dest_balance_ratio= req.newbalanceDest / (req.oldbalanceDest + 1) if req.oldbalanceDest > 0 else 0
    amount_orig_ratio = req.amount / (req.oldbalanceOrg + 1) if req.oldbalanceOrg > 0 else 0
    surp_orig         = abs(orig_diff - req.amount)
    surp_dest         = abs(dest_diff - req.amount)
    log_amount        = float(np.log1p(req.amount))

    feat_dict = {
        'step':               req.step,
        'type_enc':           type_enc,
        'amount':             req.amount,
        'log_amount':         log_amount,
        'oldbalanceOrg':      req.oldbalanceOrg,
        'newbalanceOrig':     req.newbalanceOrig,
        'oldbalanceDest':     req.oldbalanceDest,
        'newbalanceDest':     req.newbalanceDest,
        'orig_diff':          orig_diff,
        'dest_diff':          dest_diff,
        'orig_balance_ratio': orig_balance_ratio,
        'dest_balance_ratio': dest_balance_ratio,
        'amount_orig_ratio':  amount_orig_ratio,
        'surp_orig':          surp_orig,
        'surp_dest':          surp_dest,
    }
    vector = np.array([feat_dict[f] for f in features], dtype=np.float32)
    return vector, feat_dict


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "FraudShield AI API — visit /docs for Swagger UI"}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    try:
        x, feat_dict = engineer_features(req)
        prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
        is_fraud = prob >= 0.5

        if prob < 0.20:
            risk_level = "LOW"
        elif prob < 0.50:
            risk_level = "MEDIUM"
        elif prob < 0.80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return PredictionResponse(
            is_fraud=is_fraud,
            probability=round(prob, 6),
            risk_level=risk_level,
            features_used={k: round(v, 4) if isinstance(v, float) else v
                           for k, v in feat_dict.items()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
