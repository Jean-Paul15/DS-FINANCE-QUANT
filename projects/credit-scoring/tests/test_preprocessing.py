import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.preprocessing.transformers import Winsorizer


def test_winsorizer_no_nan_output():
    X = np.array([[1, 100], [2, 200], [3, 1000], [4, 4], [5, 5]])
    w = Winsorizer(limits=(0.25, 0.75))
    X_t = w.fit_transform(X)
    assert not np.isnan(X_t).any(), "Winsorizer produit des NaN"
    assert X_t.max() <= 1000, "Winsorizer ne doit pas augmenter les valeurs"


def test_winsorizer_clips_extremes():
    X = np.array([[1], [2], [3], [4], [100]])
    w = Winsorizer(limits=(0.01, 0.99))
    X_t = w.fit_transform(X)
    assert X_t.max() < 100, "La valeur extreme doit etre winsorizee"


def test_proba_to_score_range():
    from src.evaluation.validation import proba_to_score
    scores = proba_to_score(np.array([0.01, 0.189, 0.5, 0.9]))
    assert all(scores >= 0), "Score doit etre >= 0"
    assert all(scores <= 1000), "Score doit etre <= 1000"
    assert scores[0] > scores[-1], "PD basse = score haut"
