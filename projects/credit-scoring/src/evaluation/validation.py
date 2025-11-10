"""
Validation operationnelle : Leave-One-Country-Out, deciles, scorecard, IFRS 9.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from ..config import RANDOM_STATE, SCORE_BASE, PDO, BASE_PD, FIGURES_DIR
from ..preprocessing.pipeline import build_preprocessor
from .metrics import ks_statistic, gini_coefficient


def leave_one_country_out(df, features, target, preprocessor_config,
                          model_class, countries=None):
    """Validation Leave-One-Country-Out.

    Entraine sur N-1 pays, teste sur le pays restant.
    Mesure la robustesse transfrontaliere du modele.

    Parameters
    ----------
    df : DataFrame
    features : list
    target : str
    preprocessor_config : dict
    model_class : class (doit avoir fit/predict_proba)
    countries : list or None

    Returns
    -------
    DataFrame avec AUC par pays
    """
    if countries is None:
        countries = df["country"].unique().tolist()

    results = []
    for holdout_country in countries:
        # Split
        train_mask = df["country"] != holdout_country
        X_tr = df.loc[train_mask, features]
        y_tr = df.loc[train_mask, target]
        X_test = df.loc[~train_mask, features]
        y_test = df.loc[~train_mask, target]

        if len(y_test) < 10:
            continue

        # Preprocessing
        num_feats = [f for f in features if f in X_tr.select_dtypes(include=[np.number]).columns]
        cat_feats = [f for f in features if f not in num_feats]

        prep = build_preprocessor(num_feats, cat_feats, preprocessor_config)
        X_tr_p = prep.fit_transform(X_tr, y_tr)
        X_test_p = prep.transform(X_test)

        # Modele
        model = model_class()

        # Detection EBM : l'ExplainableBoostingClassifier necessite un DataFrame
        # avec noms de colonnes, pas un array numpy
        is_ebm_model = 'ExplainableBoosting' in type(model).__name__
        if is_ebm_model:
            X_tr_p_df = pd.DataFrame(X_tr_p, columns=num_feats + cat_feats)
            X_test_p_df = pd.DataFrame(X_test_p, columns=num_feats + cat_feats)
            model.fit(X_tr_p_df, y_tr)
            p_test = model.predict_proba(X_test_p_df)[:, 1]
        else:
            model.fit(X_tr_p, y_tr)
            p_test = model.predict_proba(X_test_p)[:, 1]

        auc = roc_auc_score(y_test, p_test)
        ks = ks_statistic(y_test, p_test)
        gini = gini_coefficient(y_test, p_test)

        results.append({
            "Pays": holdout_country,
            "Effectif": len(y_test),
            "Taux_defaut": y_test.mean(),
            "AUC": auc,
            "KS": ks,
            "Gini": gini,
        })

    return pd.DataFrame(results).sort_values("AUC")


def analyze_deciles(y_true, y_proba):
    """Analyse par decile : monotonie et capture des defauts.

    Returns
    -------
    DataFrame avec statistiques par decile
    """
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, "values") else y_true,
                       "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"], 10, labels=False, duplicates="drop") + 1

    stats = df.groupby("decile").agg(
        n=("y", "count"),
        n_defauts=("y", "sum"),
        taux_defaut=("y", "mean"),
        proba_moy=("proba", "mean"),
    ).reset_index()

    stats["pct_defauts"] = stats["n_defauts"] / stats["n_defauts"].sum() * 100
    stats["cum_pct"] = stats["pct_defauts"].cumsum()
    stats["bad_rate"] = stats["taux_defaut"] * 100
    return stats


def proba_to_score(p, base_score=SCORE_BASE, pdo=PDO, base_pd=BASE_PD):
    """Conversion PD -> score 0-1000. Plus haut = moins risque."""
    base_odds = base_pd / (1 - base_pd)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds = np.clip(p / (1 - p + 1e-9), 1e-9, 1e9)
    return np.clip(1000 - (offset + factor * np.log(odds)), 0, 1000)


def classify_ifrs9(proba):
    """Classification IFRS 9 selon les seuils reglementaires."""
    if proba < 0.10:
        return "Stage 1 - Performing"
    elif proba < 0.30:
        return "Stage 1+ - Surveillance"
    elif proba < 0.50:
        return "Stage 2 - Under-performing"
    else:
        return "Stage 3 - Non-performing"


def plot_decile_analysis(dec_stats, y_oot_mean, filename="decile_analysis.png"):
    """Trace et sauvegarde l'analyse par decile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(dec_stats["decile"], dec_stats["bad_rate"], color="#3498db")
    axes[0].axhline(y=y_oot_mean * 100, color="red", linestyle="--", lw=2,
                    label=f"Moyenne OOT ({y_oot_mean*100:.1f}%)")
    axes[0].set_xlabel("Decile (par PD predite)")
    axes[0].set_ylabel("Taux de defaut observe (%)")
    axes[0].set_title("Taux de defaut par decile", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(dec_stats["decile"], dec_stats["cum_pct"], "b-o", lw=2, ms=8, label="Modele")
    axes[1].plot([1, 10], [10, 100], "k--", lw=1, label="Aleatoire")
    axes[1].set_xlabel("Decile")
    axes[1].set_ylabel("% defauts cumules")
    axes[1].set_title("Courbe de gain cumule", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
