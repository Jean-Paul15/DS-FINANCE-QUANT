"""
Calibration des probabilites — Platt Scaling et Isotonic Regression.
Ajuste les probabilites predites pour qu'elles refletent
les frequences observees de defaut. Essentiel pour IFRS 9.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
    CalibrationDisplay,
)
from sklearn.metrics import brier_score_loss
from ..config import RANDOM_STATE, FIGURES_DIR


def calibrate_model(base_model, X_train, y_train, method="isotonic", cv=3):
    """Calibre un modele avec Platt Scaling ou Isotonic Regression.

    Parameters
    ----------
    base_model : classifieur fitted
        Modele de base (doit avoir predict_proba)
    X_train : array
        Donnees d'entrainement pour le calibrateur
    y_train : array
        Labels pour le calibrateur
    method : str
        "sigmoid" (Platt) ou "isotonic"
    cv : str or int
        "prefit" ou nombre de folds

    Returns
    -------
    modele calibre
    """
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compare_calibration_methods(base_model, X_train, y_train, X_val, y_val,
                                X_cal=None, y_cal=None, output_dir=FIGURES_DIR):
    """Compare Platt vs Isotonic vs non calibre.

    Utilise un 3-way split (train / calibration / test) si X_cal est fourni,
    sinon utilise X_train pour la calibration (comportement actuel).

    Parameters
    ----------
    base_model : classifieur fitted
    X_train, y_train : donnees d'entrainement
    X_val, y_val : donnees de validation/evaluation
    X_cal, y_cal : donnees de calibration (optionnel). Si fourni, utilisees
                   pour fit le calibrateur au lieu de X_train.

    Returns
    -------
    dict avec Brier scores et modeles calibres
    """
    # 3-way split : utiliser X_cal pour la calibration si fourni
    X_cal_fit = X_cal if X_cal is not None else X_train
    y_cal_fit = y_cal if y_cal is not None else y_train

    results = {}

    # Modele non calibre
    if hasattr(base_model, "predict_proba"):
        p_raw = base_model.predict_proba(X_val)[:, 1]
        brier_raw = brier_score_loss(y_val, p_raw)
    else:
        brier_raw = None
        p_raw = None

    # Platt Scaling (sigmoid)
    try:
        cal_platt = calibrate_model(base_model, X_cal_fit, y_cal_fit, method="sigmoid")
        p_platt = cal_platt.predict_proba(X_val)[:, 1]
        brier_platt = brier_score_loss(y_val, p_platt)
        results["platt"] = {"brier": brier_platt, "model": cal_platt, "probas": p_platt}
    except Exception as e:
        print(f"Platt scaling error: {e}")
        results["platt"] = {"brier": None, "model": None, "probas": None}

    # Isotonic Regression
    try:
        cal_iso = calibrate_model(base_model, X_cal_fit, y_cal_fit, method="isotonic")
        p_iso = cal_iso.predict_proba(X_val)[:, 1]
        brier_iso = brier_score_loss(y_val, p_iso)
        results["isotonic"] = {"brier": brier_iso, "model": cal_iso, "probas": p_iso}
    except Exception as e:
        print(f"Isotonic calibration error: {e}")
        results["isotonic"] = {"brier": None, "model": None, "probas": None}

    results["uncalibrated"] = {"brier": brier_raw, "model": base_model, "probas": p_raw}

    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        if res["probas"] is not None:
            prob_true, prob_pred = calibration_curve(y_val, res["probas"], n_bins=10)
            ax.plot(prob_pred, prob_true, "o-", lw=2, ms=6,
                    label=f"{name} (Brier={res['brier']:.4f})" if res["brier"] else name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Parfait")
    ax.set_xlabel("Probabilite predite moyenne")
    ax.set_ylabel("Fraction de defauts observee")
    ax.set_title("Courbes de calibration — Platt vs Isotonic vs Brut", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def calibration_report(y_true, y_proba, n_bins=10):
    """Rapport detaille de calibration par bin.

    Returns
    -------
    DataFrame avec ecarts par bin
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    report = df.groupby("bin", observed=True).agg(
        n=("y", "count"),
        proba_moy=("p", "mean"),
        taux_observe=("y", "mean"),
    ).reset_index()
    report["ecart_bps"] = (report["proba_moy"] - report["taux_observe"]) * 10000  # bps
    report["brier_bin"] = (report["ecart_bps"] / 10000) ** 2
    return report
