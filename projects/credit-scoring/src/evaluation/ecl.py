"""
ECL - Expected Credit Loss IFRS 9
==================================
Calcule les provisions ECL = PD x LGD x EAD x DF
Avec support des 3 stages IFRS 9 et des scenarios ponderes.
"""
import numpy as np
import pandas as pd


def calculate_lgd(df, method="collateral_based"):
    """Calcule la Loss Given Default (LGD).

    Parameters
    ----------
    df : DataFrame avec has_collateral, collateral_value, loan_amount, loan_to_value
    method : str
        "collateral_based" : LGD basee sur la valeur de la garantie
        "regulatory" : LGD forfaitaire BCEAO (45% non-securise, 35% securise)

    Returns
    -------
    Series LGD (entre 0 et 1)

    Notes
    -----
    Formule: LGD = 1 - Recovery Rate
    Recovery Rate = min(collateral_value / exposure, 1) x haircut
    Haircut = 0.50 pour inventory, 0.30 pour real_estate, 0.40 pour vehicle, 0.60 pour equipment
    """
    df = df.copy()

    if method == "regulatory":
        return pd.Series(np.where(df["has_collateral"] == 1, 0.35, 0.45), index=df.index)

    # Collateral-based LGD
    haircuts = {
        "Inventory": 0.50, "Real_estate": 0.30, "Vehicle": 0.40,
        "Equipment": 0.60, "Cash_deposit": 0.10, None: 0.80
    }

    # Taux de recouvrement base sur la garantie
    recovery = np.zeros(len(df))
    for i in range(len(df)):
        collat_val = df["collateral_value"].iloc[i] if pd.notna(df["collateral_value"].iloc[i]) else 0
        exposure = df["loan_amount"].iloc[i]
        collat_type = df["collateral_type"].iloc[i] if pd.notna(df["collateral_type"].iloc[i]) else None

        haircut = haircuts.get(collat_type, 0.80)

        if exposure > 0 and collat_val > 0:
            coverage_ratio = min(collat_val / exposure, 1.0)
            recovery[i] = coverage_ratio * (1 - haircut)
        # else: recovery reste 0

    # LGD = 1 - Recovery, avec plancher reglementaire BCEAO de 10%
    lgd = 1 - recovery
    lgd = np.clip(lgd, 0.10, 0.90)

    # Si pas de garantie, LGD = 0.45 (standard BCEAO non-securise)
    no_collateral_mask = (df["has_collateral"] == 0) | (df["collateral_value"].fillna(0) == 0)
    lgd[no_collateral_mask] = 0.45

    return pd.Series(lgd, index=df.index)


def calculate_ead(df):
    """Calcule l'Exposure at Default (EAD).

    Pour les prets a terme (fixe) : EAD = encours restant du.
    Pour les lignes de credit : EAD = tirage + CCF x (ligne - tirage)

    Ici, on suppose un pret a terme avec amortissement lineaire.
    On estime que 50% du capital a ete rembourse au moment du defaut (maturite moyenne).

    Returns
    -------
    Series EAD
    """
    df = df.copy()

    # EAD = solde restant du estime au moment du defaut
    # Hypothese: defaut a mi-vie du pret -> 50% du capital rembourse
    ead = df["loan_amount"].fillna(0) * 0.50

    # Ajustement pour prets avec garantie: l'EAD est reduit du montant garanti
    # (mais cela est deja capture dans la LGD)

    return ead.clip(lower=0)


def calculate_discount_factor(term_months, rate=0.05):
    """Calcule le facteur d'actualisation pour l'ECL.

    DF = 1 / (1 + r)^(T/12) ou T est la maturite restante estimee.
    """
    return 1 / (1 + rate) ** (term_months / 12)


def calculate_ecl(df, pd_values, lgd_values=None, ead_values=None,
                  discount_rate=0.05, stage_col="stage"):
    """Calcule l'ECL (Expected Credit Loss) IFRS 9.

    ECL = PD x LGD x EAD x DF

    Parameters
    ----------
    df : DataFrame
    pd_values : array PD (12 mois ou lifetime)
    lgd_values : array LGD (optionnel, calcule si None)
    ead_values : array EAD (optionnel, calcule si None)
    discount_rate : float
    stage_col : str, colonne de stage IFRS 9 (si presente dans df)

    Returns
    -------
    DataFrame avec colonnes: PD, LGD, EAD, DF, ECL, ECL_stage
    """
    df = df.copy()

    result = pd.DataFrame(index=df.index)
    result["PD"] = pd_values

    if lgd_values is None:
        result["LGD"] = calculate_lgd(df, method="collateral_based")
    else:
        result["LGD"] = lgd_values

    if ead_values is None:
        result["EAD"] = calculate_ead(df)
    else:
        result["EAD"] = ead_values

    result["DF"] = calculate_discount_factor(df["term_months"].fillna(24), discount_rate)
    result["ECL"] = result["PD"] * result["LGD"] * result["EAD"] * result["DF"]

    # Segmentation IFRS 9
    if stage_col in df.columns:
        result["stage"] = df[stage_col]
    else:
        # Calculer le stage depuis PD
        def _stage(p):
            if p < 0.10: return "Stage 1"
            elif p < 0.30: return "Stage 1+"
            elif p < 0.50: return "Stage 2"
            return "Stage 3"
        result["stage"] = pd.Series(pd_values).apply(_stage)

    # Resume par stage
    ecl_summary = result.groupby("stage").agg(
        n=("ECL", "count"),
        ECL_totale=("ECL", "sum"),
        ECL_moyenne=("ECL", "mean"),
        PD_moyenne=("PD", "mean"),
        LGD_moyenne=("LGD", "mean"),
        EAD_moyenne=("EAD", "mean"),
    ).round(4)

    return result, ecl_summary


def calculate_scenario_weighted_ecl(df, pd_baseline, pd_adverse, pd_severe,
                                     weights=None, lgd_values=None, ead_values=None):
    """ECL ponderee par scenario IFRS 9.

    ECL_final = w_baseline x ECL_baseline + w_adverse x ECL_adverse + w_severe x ECL_severe
    """
    if weights is None:
        weights = {"baseline": 0.60, "adverse": 0.30, "severe": 0.10}

    ecl_baseline, _ = calculate_ecl(df, pd_baseline, lgd_values, ead_values)
    ecl_adverse, _ = calculate_ecl(df, pd_adverse, lgd_values, ead_values)
    ecl_severe, _ = calculate_ecl(df, pd_severe, lgd_values, ead_values)

    ecl_weighted = (
        weights["baseline"] * ecl_baseline["ECL"] +
        weights["adverse"] * ecl_adverse["ECL"] +
        weights["severe"] * ecl_severe["ECL"]
    )

    result = pd.DataFrame({
        "ECL_baseline": ecl_baseline["ECL"],
        "ECL_adverse": ecl_adverse["ECL"],
        "ECL_severe": ecl_severe["ECL"],
        "ECL_weighted": ecl_weighted,
        "PD_baseline": pd_baseline,
        "PD_adverse": pd_adverse,
        "PD_severe": pd_severe,
        "LGD": ecl_baseline["LGD"],
        "EAD": ecl_baseline["EAD"],
    })

    summary = {
        "ECL_baseline_total": float(ecl_baseline["ECL"].sum()),
        "ECL_adverse_total": float(ecl_adverse["ECL"].sum()),
        "ECL_severe_total": float(ecl_severe["ECL"].sum()),
        "ECL_weighted_total": float(ecl_weighted.sum()),
        "ECL_moyen_par_client": float(ecl_weighted.mean()),
        "poids_scenarios": weights,
    }

    return result, summary


def lifetime_pd_curve(pd_12m, maturity_months, method="exponential"):
    """Estime la courbe de PD vie entiere a partir de la PD 12 mois.

    Parameters
    ----------
    pd_12m : array, PD a 12 mois
    maturity_months : array, maturite restante en mois
    method : str
        "exponential" : PD(t) = 1 - (1 - PD_12m)^(t/12)
        "linear" : PD(t) = PD_12m x t/12 (approximation conservative)

    Returns
    -------
    array, PD cumulee a maturite
    """
    t = np.asarray(maturity_months) / 12  # en annees
    t = np.clip(t, 0.25, 10)  # entre 3 mois et 10 ans

    if method == "exponential":
        return 1 - (1 - np.asarray(pd_12m)) ** t
    else:
        return np.asarray(pd_12m) * t
