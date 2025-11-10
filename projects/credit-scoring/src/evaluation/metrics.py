"""
Metriques d'evaluation avancees : PSI, stabilite, calibration.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def population_stability_index(expected_probas, actual_probas, bins=10):
    """Calcule le Population Stability Index (PSI).

    Mesure la deviation entre les distributions de probabilites
    de deux populations (ex: train vs OOT).

    Interpretation :
    - PSI < 0.10 : Stable
    - 0.10 <= PSI < 0.25 : Deviation moderee (surveiller)
    - PSI >= 0.25 : Deviation significative (recalibrer)

    Parameters
    ----------
    expected_probas : array
        Probas de reference (ex: train)
    actual_probas : array
        Probas a comparer (ex: OOT)
    bins : int
        Nombre de bins pour discretiser

    Returns
    -------
    float PSI
    """
    expected = np.asarray(expected_probas)
    actual = np.asarray(actual_probas)
    eps = 1e-10

    breakpoints = np.linspace(0, 1, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def feature_psi(train_values, oot_values, feature_name, bins=10):
    """PSI par feature entre train et OOT.

    Parameters
    ----------
    train_values : Series
    oot_values : Series
    feature_name : str
    bins : int

    Returns
    -------
    dict {feature, psi, interpretation}
    """
    train_vals = train_values.dropna()
    oot_vals = oot_values.dropna()

    if len(train_vals) < 10 or len(oot_vals) < 10:
        return {"feature": feature_name, "psi": np.nan, "interpretation": "Donnees insuffisantes"}

    # Bin edges based on train distribution
    if train_vals.nunique() < bins:
        bin_edges = np.sort(train_vals.unique())
    else:
        bin_edges = np.percentile(train_vals.values, np.linspace(0, 100, bins + 1))

    eps = 1e-10
    train_counts, _ = np.histogram(train_vals.values, bins=bin_edges)
    oot_counts, _ = np.histogram(oot_vals.values, bins=bin_edges)

    train_pct = train_counts / len(train_vals) + eps
    oot_pct = oot_counts / len(oot_vals) + eps

    psi = float(np.sum((oot_pct - train_pct) * np.log(oot_pct / train_pct)))

    if psi < 0.10:
        interp = "Stable"
    elif psi < 0.25:
        interp = "Deviation moderee"
    else:
        interp = "Shift significatif"

    return {"feature": feature_name, "psi": psi, "interpretation": interp}


def calculate_all_psi(df_train, df_oot, features):
    """Calcule le PSI pour chaque feature.

    Returns
    -------
    DataFrame trie par PSI decroissant
    """
    results = []
    for f in features:
        if f in df_train.columns and f in df_oot.columns:
            r = feature_psi(df_train[f], df_oot[f], f)
            results.append(r)

    return pd.DataFrame(results).sort_values("psi", ascending=False)


def ks_statistic(y_true, y_proba):
    """Calcule le Kolmogorov-Smirnov statistic."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float((tpr - fpr).max())


def gini_coefficient(y_true, y_proba):
    """Gini = 2 * AUC - 1."""
    auc = roc_auc_score(y_true, y_proba)
    return 2 * auc - 1


def calibration_report(y_true, y_proba, n_bins=10):
    """Rapport de calibration detaille par bin.

    Returns
    -------
    DataFrame avec proba_moy, taux_observe, ecart
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    report = df.groupby("bin", observed=True).agg(
        n=("y", "count"),
        proba_moy=("p", "mean"),
        taux_observe=("y", "mean"),
    ).reset_index()
    report["ecart"] = report["proba_moy"] - report["taux_observe"]
    report["brier_bin"] = (report["ecart"]) ** 2
    return report
