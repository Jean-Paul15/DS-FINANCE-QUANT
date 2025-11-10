"""
Benchmark de modeles avec bootstrap confidence intervals.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from ..config import RANDOM_STATE, BOOTSTRAP_ITER


def get_models(scale_pos_weight=1.0):
    """Retourne les 6 modeles a benchmarker.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio neg/pos pour XGBoost
    """
    return {
        "LogReg_L2": LogisticRegression(
            C=5.0, penalty="l2", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "LogReg_L1": LogisticRegression(
            C=5.0, penalty="l1", solver="saga", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, subsample=0.8, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="auc",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_child_samples=20, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


def train_and_evaluate(model, name, X_tr, y_tr, X_val, y_val, X_oot, y_oot,
                       feature_names=None, n_bootstrap=BOOTSTRAP_ITER):
    """Entraine un modele et calcule toutes les metriques avec bootstrap CI.

    Returns
    -------
    dict metriques, modele fitted, dict probas
    """
    is_ebm = isinstance(model, ExplainableBoostingClassifier)

    if is_ebm:
        X_tr_df = pd.DataFrame(X_tr, columns=feature_names) if not isinstance(X_tr, pd.DataFrame) else X_tr
        X_val_df = pd.DataFrame(X_val, columns=feature_names) if not isinstance(X_val, pd.DataFrame) else X_val
        X_oot_df = pd.DataFrame(X_oot, columns=feature_names) if not isinstance(X_oot, pd.DataFrame) else X_oot
        model.fit(X_tr_df, y_tr)
        p_tr = model.predict_proba(X_tr_df)[:, 1]
        p_val = model.predict_proba(X_val_df)[:, 1]
        p_oot = model.predict_proba(X_oot_df)[:, 1]
    else:
        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        p_oot = model.predict_proba(X_oot)[:, 1]

    auc_oot = roc_auc_score(y_oot, p_oot)
    fpr, tpr, _ = roc_curve(y_oot, p_oot)
    ks = (tpr - fpr).max()

    # Bootstrap CI
    boots = []
    rng = np.random.RandomState(RANDOM_STATE)
    y_oot_arr = y_oot.values if hasattr(y_oot, "values") else y_oot
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_oot_arr), len(y_oot_arr), replace=True)
        try:
            boots.append(roc_auc_score(y_oot_arr[idx], p_oot[idx]))
        except Exception:
            pass

    ci_low = np.percentile(boots, 2.5)
    ci_high = np.percentile(boots, 97.5)

    return (
        {
            "modele": name,
            "auc_train": roc_auc_score(y_tr, p_tr),
            "auc_val": roc_auc_score(y_val, p_val),
            "auc_oot": auc_oot,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "gini_oot": 2 * auc_oot - 1,
            "ks_oot": ks,
            "brier_oot": brier_score_loss(y_oot, p_oot),
            "logloss_oot": log_loss(y_oot, p_oot),
        },
        model,
        {"train": p_tr, "val": p_val, "oot": p_oot},
    )


def benchmark_all(models_dict, X_tr, y_tr, X_val, y_val, X_oot, y_oot,
                  feature_names=None):
    """Execute le benchmark complet.

    Returns
    -------
    df_results, trained_models, all_probas
    """
    results = []
    trained = {}
    probas = {}

    for name, model in models_dict.items():
        res, fitted, prob = train_and_evaluate(
            model, name, X_tr, y_tr, X_val, y_val, X_oot, y_oot,
            feature_names=feature_names,
        )
        res["overfit"] = res["auc_train"] - res["auc_val"]
        res["stabilite"] = abs(res["auc_val"] - res["auc_oot"])
        results.append(res)
        trained[name] = fitted
        probas[name] = prob

    df_results = pd.DataFrame(results).sort_values("auc_oot", ascending=False)
    return df_results, trained, probas


def train_ebm(X_tr, y_tr, X_val, y_val, X_oot, y_oot, feature_names=None,
              interactions=5, learning_rate=0.02):
    """Entraine un EBM dedie (pour interpretabilite)."""
    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        learning_rate=learning_rate,
        max_bins=64,
        max_interaction_bins=32,
        random_state=RANDOM_STATE,
    )

    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_oot_df = pd.DataFrame(X_oot, columns=feature_names)

    ebm.fit(X_tr_df, y_tr)
    p_oot = ebm.predict_proba(X_oot_df)[:, 1]

    return ebm, roc_auc_score(y_oot, p_oot), p_oot
