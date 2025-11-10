"""
Configuration centrale du projet Credit Scoring UCB.
"""
from pathlib import Path

# Chemins
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
OUTPUTS = ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
FIGURES_DIR = OUTPUTS / "figures"
NOTEBOOKS = ROOT / "notebooks"

for d in [OUTPUTS, MODELS_DIR, FIGURES_DIR, DATA_RAW, DATA_EXT]:
    d.mkdir(parents=True, exist_ok=True)

# Parametres modelisation
RANDOM_STATE = 42
TARGET = "default"
DATE_CUTOFF = "2024-07-01"

# IV
IV_THRESHOLDS = {
    "inutile": 0.02,
    "faible": 0.10,
    "moyen": 0.30,
    "fort": 0.50,
}

# Preprocessing
IMPUTATION_METHODS = ["median", "mean", "iterative", "knn"]
SCALING_METHODS = ["standard", "robust", "minmax", "quantile"]
WINSOR_LIMITS = (0.01, 0.99)
CORRELATION_THRESHOLD = 0.80
VIF_THRESHOLD = 10

# Feature selection
N_FEATURES_MAX = 35
N_FEATURES_MIN = 20
CONSENSUS_VOTES = 2

# Modeles
CV_FOLDS = 5
NESTED_CV_OUTER = 3
NESTED_CV_INNER = 3
BOOTSTRAP_ITER = 500
CONFIDENCE_LEVEL = 0.95

# Scoring
SCORE_BASE = 500
PDO = 20
BASE_PD = 0.189

# Interpretabilite
SHAP_SAMPLE = 300
LIME_FEATURES = 8
EBM_INTERACTIONS = 5

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
