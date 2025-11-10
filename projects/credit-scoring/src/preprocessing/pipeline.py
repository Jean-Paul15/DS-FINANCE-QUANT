"""
Pipeline de preprocessing avec strategies configurables.
Systematiquement teste Median, Mean, Iterative, KNN, MissForest
avec StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
et Winsorization optionnelle.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .transformers import Winsorizer, MissForestImputer, TO_OBJECT_TRANSFORMER
from ..config import WINSOR_LIMITS, RANDOM_STATE


def build_numeric_pipeline(
    imputer_type="median",
    scaler_type="robust",
    winsorize=False,
    winsor_limits=WINSOR_LIMITS,
    use_power_transform=False,
):
    """Construit le pipeline de preprocessing numerique.

    Parameters
    ----------
    imputer_type : str
        "median", "mean", "iterative", "knn", "missforest"
    scaler_type : str
        "standard", "robust", "minmax", "quantile"
    winsorize : bool
        Appliquer winsorization avant imputation
    winsor_limits : tuple
        (low_percentile, high_percentile)
    use_power_transform : bool
        Appliquer PowerTransformer (Yeo-Johnson) apres imputation

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []

    if winsorize:
        steps.append(("winsor", Winsorizer(winsor_limits)))

    if imputer_type == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputer_type == "mean":
        steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif imputer_type == "iterative":
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        steps.append(("imputer", IterativeImputer(max_iter=10, random_state=RANDOM_STATE)))
    elif imputer_type == "knn":
        steps.append(("imputer", KNNImputer(n_neighbors=5)))
    elif imputer_type == "missforest":
        steps.append(("imputer", MissForestImputer(n_estimators=50, max_iter=5, random_state=RANDOM_STATE)))

    if use_power_transform:
        steps.append(("power", PowerTransformer(method="yeo-johnson")))

    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler_type == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_type == "quantile":
        steps.append(("scaler", QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)))

    return Pipeline(steps)


def build_categorical_pipeline():
    """Pipeline categoriel standard."""
    return Pipeline([
        ("to_object", TO_OBJECT_TRANSFORMER),
        ("imputer", SimpleImputer(strategy="constant", fill_value="MANQUANT")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])


def build_preprocessor(num_features, cat_features, num_config=None):
    """Construit le preprocessor complet (numerique + categoriel).

    Parameters
    ----------
    num_features : list
        Noms des colonnes numeriques
    cat_features : list
        Noms des colonnes categorielles
    num_config : dict or None
        Configuration pour build_numeric_pipeline()

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    if num_config is None:
        num_config = {"imputer_type": "median", "scaler_type": "robust", "winsorize": False}

    num_pipe = build_numeric_pipeline(**num_config)
    cat_pipe = build_categorical_pipeline()

    return ColumnTransformer(
        [("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)],
        remainder="drop",
    )


def evaluate_preprocessing_strategy(
    prep, X_train, y_train, X_val, y_val, model=None
):
    """Evalue une strategie de preprocessing avec LogReg.

    Returns
    -------
    dict avec auc_train, auc_val, overfit
    """
    if model is None:
        model = LogisticRegression(
            C=0.5, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )

    X_tr_p = prep.fit_transform(X_train, y_train)
    X_val_p = prep.transform(X_val)

    model.fit(X_tr_p, y_train)
    p_tr = model.predict_proba(X_tr_p)[:, 1]
    p_val = model.predict_proba(X_val_p)[:, 1]

    return {
        "auc_train": roc_auc_score(y_train, p_tr),
        "auc_val": roc_auc_score(y_val, p_val),
        "overfit": roc_auc_score(y_train, p_tr) - roc_auc_score(y_val, p_val),
    }


def generate_strategy_configs():
    """Genere les configurations de strategies a tester.

    Returns
    -------
    list of (name, config_dict)
    """
    configs = []

    # Strategies de base (rapides)
    for imp in ["median", "mean"]:
        for scl in ["standard", "robust"]:
            for win in [False, True]:
                name = f"{imp}_{scl}"
                if win:
                    name += "_win"
                configs.append((name, {
                    "imputer_type": imp,
                    "scaler_type": scl,
                    "winsorize": win,
                }))

    # Strategies avancees
    configs += [
        ("iterative_standard_win", {
            "imputer_type": "iterative", "scaler_type": "standard", "winsorize": True
        }),
        ("iterative_robust_win", {
            "imputer_type": "iterative", "scaler_type": "robust", "winsorize": True
        }),
        ("knn_robust_win", {
            "imputer_type": "knn", "scaler_type": "robust", "winsorize": True
        }),
        ("median_quantile_win", {
            "imputer_type": "median", "scaler_type": "quantile", "winsorize": True
        }),
    ]

    return configs
