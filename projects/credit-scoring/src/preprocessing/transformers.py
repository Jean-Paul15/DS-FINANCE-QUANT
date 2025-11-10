"""
Transformers personnalises pour le preprocessing.
Winsorizer, MissForest wrapper, convertisseurs.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limite les valeurs extremes aux percentiles specifies.

    Parameters
    ----------
    limits : tuple (low, high), default (0.01, 0.99)
        Percentiles de winsorization.
    """
    def __init__(self, limits=(0.01, 0.99)):
        self.limits = limits

    def fit(self, X, y=None):
        self.lower_ = np.nanpercentile(X, self.limits[0] * 100, axis=0)
        self.upper_ = np.nanpercentile(X, self.limits[1] * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


class MissForestImputer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compatible pour MissForest (missingpy).

    Utilise Random Forest pour imputer les valeurs manquantes
    de facon iterative, en capturant les interactions non-lineaires.
    """
    def __init__(self, n_estimators=50, max_iter=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        from missingpy import MissForest
        # MissForest est un transformeur "fit_transform" seulement
        self.mf_ = MissForest(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            random_state=self.random_state,
            copy=False
        )
        # On conserve les colonnes pour la transformation
        self.columns_ = list(range(X.shape[1]))
        return self

    def transform(self, X):
        from missingpy import MissForest
        mf = MissForest(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            random_state=self.random_state,
            copy=False
        )
        return mf.fit_transform(X)


class DistributionChecker(BaseEstimator, TransformerMixin):
    """Verifie les distributions avant/apres transformation.

    Calcule skewness, kurtosis, et % d'outliers pour chaque colonne.
    Utile pour documenter l'impact de chaque etape de preprocessing.
    """
    def __init__(self, stage="after"):
        self.stage = stage

    def fit(self, X, y=None):
        self.stats_ = {}
        X_arr = np.asarray(X)
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) > 0:
                self.stats_[j] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "skewness": float(pd.Series(finite).skew()),
                    "kurtosis": float(pd.Series(finite).kurtosis()),
                    "pct_outliers_iqr": float(
                        np.mean((finite < np.percentile(finite, 25) - 1.5 * (np.percentile(finite, 75) - np.percentile(finite, 25))) |
                                (finite > np.percentile(finite, 75) + 1.5 * (np.percentile(finite, 75) - np.percentile(finite, 25))))
                    ) * 100
                }
        return self

    def transform(self, X):
        return X

    def get_stats_dataframe(self, feature_names=None):
        """Retourne les statistiques sous forme de DataFrame."""
        rows = []
        for j, stats in self.stats_.items():
            row = {"feature": feature_names[j] if feature_names else j}
            row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)


def _to_object(X):
    """Convertit StringDtype pandas -> object numpy (compatibilite sklearn)."""
    return X.astype(object)


TO_OBJECT_TRANSFORMER = FunctionTransformer(
    _to_object, feature_names_out="one-to-one"
)
