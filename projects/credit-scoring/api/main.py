"""
API REST — Credit Scoring UCB
==============================
FastAPI pour l'inference en production.

Endpoints:
  POST /score          Scoring complet (PD + score + IFRS 9)
  POST /explain        Explication individuelle (EBM only)
  GET  /health         Healthcheck
  GET  /metadata       Informations sur le modele
"""
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import joblib
import json
import logging
import os
from pathlib import Path
from datetime import datetime

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des fichiers modele
ROOT = Path(__file__).resolve().parent.parent
# Dossier des artefacts (modele, preprocesseur, features) : configurable par variable d'environnement
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "artifacts")))
MODEL_PATH = MODEL_DIR / "model.joblib"
PREP_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURES_PATH = MODEL_DIR / "features.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Chargement au demarrage
model = None
preprocessor = None
selected_features = None
metadata = None

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREP_PATH)
    features = joblib.load(FEATURES_PATH)
    if "selected" in features:
        selected_features = features["selected"]
    elif "all" in features:
        selected_features = features["all"]
    else:
        selected_features = features.get("selected_num", []) + features.get("selected_cat", [])

    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

    is_ebm = "ExplainableBoosting" in type(model).__name__
    logger.info(f"Modele charge : {type(model).__name__}")
    logger.info(f"Features : {len(selected_features)}")
    logger.info(f"EBM : {is_ebm}")
except Exception as e:
    logger.error(f"Erreur chargement modele : {e}")

# Import du feature engineering
import sys
sys.path.insert(0, str(ROOT))
from src.features.engineering import feature_engineering
from src.evaluation.validation import proba_to_score, classify_ifrs9

# FastAPI app
app = FastAPI(
    title="Credit Scoring API — UCB",
    description="Modele de probabilite de defaut IFRS 9 / BCEAO",
    version="5.0.0",
)


class ClientData(BaseModel):
    """Donnees d'un client pour scoring."""
    customer_id: Optional[str] = None
    # Demographiques
    age: int
    country: str
    gender: str
    marital_status: Optional[str] = None
    education: Optional[str] = None
    employment_type: str
    sector: str
    dependents: int = 0
    # Revenus
    income_monthly: Optional[float] = None
    income_annual: Optional[float] = None
    expenses_monthly: Optional[float] = None
    # Endettement
    existing_debt: Optional[float] = None
    monthly_debt_payment: Optional[float] = None
    # Pret
    loan_amount: float
    loan_purpose: str
    term_months: int
    interest_rate: float
    monthly_payment: Optional[float] = None
    has_collateral: int = 0
    collateral_type: Optional[str] = None
    collateral_value: Optional[float] = None
    loan_to_value: Optional[float] = None
    # Actifs
    total_assets: Optional[float] = None
    savings_balance: Optional[float] = None
    checking_avg_balance: Optional[float] = None
    # Bureau
    bureau_score: Optional[float] = None
    inquiries_6m: Optional[float] = None
    inquiries_3m: Optional[float] = None
    total_credit_lines: Optional[float] = None
    open_credit_lines: Optional[float] = None
    credit_utilization: Optional[float] = None
    derogatory_marks: Optional[float] = None
    oldest_credit_months: Optional[float] = None
    avg_credit_age_months: Optional[float] = None
    bureau_data_available: bool = True
    # Paiement
    dpd_max_24m: Optional[float] = None
    n_dpd_30_12m: Optional[float] = None
    bounced_checks_12m: Optional[float] = None
    overdrafts_12m: Optional[float] = None
    # Bancaire
    tenure_months: Optional[float] = None
    n_products: Optional[int] = 1
    has_savings_account: int = 0
    has_insurance: int = 0
    transaction_freq_monthly: Optional[int] = None
    avg_transaction_amount: Optional[float] = None
    total_transactions_12m: Optional[int] = None
    uses_mobile_banking: int = 0
    email_verified: int = 0
    years_employed: Optional[float] = None


class ScoringResponse(BaseModel):
    customer_id: str
    pd: float
    score: int
    ifrs9_stage: str


class ExplanationResponse(BaseModel):
    customer_id: str
    pd: float
    contributions: List[Dict[str, float]]


@app.on_event("startup")
async def startup_event():
    if model is None:
        logger.warning("Modele non charge ! Verifiez les fichiers .joblib")


@app.get("/health")
async def health():
    """Endpoint de healthcheck."""
    return {
        "status": "ok" if model is not None else "degrade",
        "model": type(model).__name__ if model else "non charge",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metadata")
async def get_metadata():
    """Retourne les metadonnees du modele."""
    if metadata is None:
        raise HTTPException(status_code=404, detail="Metadata non disponible")
    return metadata


@app.post("/score", response_model=ScoringResponse)
async def score_client(data: ClientData):
    """Score un client et retourne PD, score, et classification IFRS 9."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        df = pd.DataFrame([data.model_dump()])
        if data.customer_id is None:
            df["customer_id"] = "API_REQUEST"

        # Feature engineering
        df_fe = feature_engineering(df)

        # Selection features manquantes
        missing_feats = [f for f in selected_features if f not in df_fe.columns]
        if missing_feats:
            logger.warning(f"Features manquantes : {missing_feats}")

        X = df_fe[[f for f in selected_features if f in df_fe.columns]]

        # Preprocessing + prediction
        X_proc = preprocessor.transform(X)
        if is_ebm:
            X_proc_df = pd.DataFrame(X_proc, columns=[f for f in selected_features if f in df_fe.columns])
            proba = float(model.predict_proba(X_proc_df)[0, 1])
        else:
            proba = float(model.predict_proba(X_proc)[0, 1])

        score = int(proba_to_score(np.array([proba]))[0])
        stage = classify_ifrs9(proba)

        return ScoringResponse(
            customer_id=data.customer_id or "API_REQUEST",
            pd=round(proba, 4),
            score=score,
            ifrs9_stage=stage,
        )
    except Exception as e:
        logger.error(f"Erreur scoring : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain_client(data: ClientData):
    """Explication individuelle (disponible uniquement pour EBM)."""
    if not is_ebm:
        raise HTTPException(status_code=400, detail="Explication disponible uniquement pour EBM")

    try:
        df = pd.DataFrame([data.model_dump()])
        if data.customer_id is None:
            df["customer_id"] = "API_REQUEST"

        df_fe = feature_engineering(df)
        X = df_fe[[f for f in selected_features if f in df_fe.columns]]
        X_proc = preprocessor.transform(X)
        X_proc_df = pd.DataFrame(X_proc, columns=[f for f in selected_features if f in df_fe.columns])

        proba = float(model.predict_proba(X_proc_df)[0, 1])
        local_exp = model.explain_local(X_proc_df)
        exp_data = local_exp.data(0)

        contributions = [
            {"feature": name, "contribution": float(score)}
            for name, score in zip(exp_data["names"], exp_data["scores"])
        ]
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return ExplanationResponse(
            customer_id=data.customer_id or "API_REQUEST",
            pd=round(proba, 4),
            contributions=contributions[:10],
        )
    except Exception as e:
        logger.error(f"Erreur explication : {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
