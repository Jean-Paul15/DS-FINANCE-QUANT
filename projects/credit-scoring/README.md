# Scoring de crédit IFRS 9 (UEMOA)

Modèle de probabilité de défaut à 12 mois sur données synthétiques, avec classification en stages IFRS 9, calcul d'ECL et API de scoring.

## Contenu

- `src/data/` : générateur de données (`generator.py`) et variables macroéconomiques (`macro.py`).
- `src/preprocessing/` : pipeline sklearn, transformeurs (winsorisation), WoE, reject inference.
- `src/features/` : feature engineering et sélection (IV, corrélation, LightGBM, RFE, Boruta).
- `src/models/benchmark.py` : benchmark de modèles (régression logistique, random forest, XGBoost, LightGBM, EBM).
- `src/evaluation/` : métriques, calibration, validation, ECL (PD x LGD x EAD x facteur d'actualisation).
- `src/interpretability/` : SHAP, LIME, fonctions de forme EBM.
- `master_pipeline.py` : exécution séquentielle de l'ensemble (EDA, preprocessing, sélection, validation croisée, benchmark, PSI, calibration, ECL).
- `inference.py` : classe `CreditScorer` (PD, score 0-1000, classe IFRS 9, explication, ECL).
- `api/main.py` : API FastAPI (`/score`, `/explain`, `/health`, `/metadata`).
- `data/external/macro_uemoa.csv` : petit échantillon macro synthétique (5 Ko).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/data/generator.py     # écrit data/raw/ et data/external/
python master_pipeline.py        # écrit outputs/ et copie les artefacts dans artifacts/
```

Inférence :

```python
from inference import CreditScorer
import pandas as pd

scorer = CreditScorer(
    model_path="artifacts/model.joblib",
    preprocessor_path="artifacts/preprocessor.joblib",
    features_path="artifacts/features.joblib",
    metadata_path="artifacts/metadata.json",
)
print(scorer.predict_full(pd.read_csv("nouveaux_clients.csv")))
```

API (les artefacts sont lus dans `MODEL_DIR`, par défaut `artifacts/`) :

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
# ou : docker compose up --build   (monte ./artifacts en lecture seule)
```

## Tests

```bash
python -m pytest tests
```

## Limites

Données synthétiques uniquement ; les chiffres de performance dépendent du générateur. Boruta et une partie de la validation croisée imbriquée sont allégées dans `master_pipeline.py` (commentaires dans le code).
