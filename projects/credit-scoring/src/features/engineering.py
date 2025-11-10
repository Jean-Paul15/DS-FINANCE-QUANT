"""
Feature Engineering cible pour scoring credit UEMOA.
Chaque feature creee est justifiee par une analyse EDA prealable.
"""
import numpy as np
import pandas as pd


def create_bureau_indicators(df):
    """Binning du bureau score — la variable la plus predictive (IV fort)."""
    df = df.copy()
    bureau = df["bureau_score"].fillna(620)
    df["bureau_low"] = (bureau < 550).astype(int)
    df["bureau_medium"] = ((bureau >= 550) & (bureau < 700)).astype(int)
    df["bureau_high"] = (bureau >= 700).astype(int)
    df["bureau_missing"] = df["bureau_score"].isna().astype(int)
    return df


def create_payment_severity(df):
    """Score composite de severite des incidents de paiement."""
    df = df.copy()
    df["dpd_severity"] = (
        df["dpd_max_24m"].fillna(0) / 30
        + df["n_dpd_30_12m"].fillna(0) * 0.5
        + df["bounced_checks_12m"].fillna(0) * 0.3
    )
    df["has_payment_issues"] = (
        (df["dpd_max_24m"].fillna(0) > 0)
        | (df["bounced_checks_12m"].fillna(0) > 0)
    ).astype(int)
    return df


def create_repayment_capacity(df, train_stats=None):
    """Indicateurs de capacite de remboursement.

    ATTENTION: les medianes sont calculees sur le DataFrame passe en argument.
    En phase d'entrainement, ce DOIT etre df_train UNIQUEMENT (pas df_raw global).
    En phase d'inference, les medianes de train doivent etre sauvegardees et reutilisees.

    Parameters
    ----------
    df : DataFrame
    train_stats : dict or None
        Si fourni, contient les medianes calculees sur le train (inc_med, exp_med).
        Necessaire en inference mono-client pour eviter median() = valeur du client.
    """
    df = df.copy()
    if train_stats:
        inc_med = train_stats["inc_med"]
        exp_med = train_stats["exp_med"]
    else:
        inc_med = df["income_monthly"].median() if df["income_monthly"].notna().any() else 250000
        exp_med = df["expenses_monthly"].median() if df["expenses_monthly"].notna().any() else 100000

    df["net_income"] = df["income_monthly"].fillna(inc_med) - df["expenses_monthly"].fillna(exp_med)
    df["repayment_capacity"] = (
        df["net_income"]
        - df["monthly_payment"].fillna(0)
        - df["monthly_debt_payment"].fillna(0)
    )
    df["repayment_ratio"] = df["repayment_capacity"] / (df["income_monthly"].fillna(inc_med) + 1)
    df["total_burden"] = (
        df["monthly_payment"].fillna(0) + df["monthly_debt_payment"].fillna(0)
    ) / (df["income_monthly"].fillna(inc_med) + 1)
    df["payment_pressure"] = df["monthly_payment"].fillna(0) / df["net_income"].clip(lower=1)
    return df


def create_stability_engagement(df):
    """Indicateurs de stabilite et engagement client."""
    df = df.copy()
    df["stable_employment"] = (df["years_employed"].fillna(0) >= 3).astype(int)
    df["long_tenure"] = (df["tenure_months"].fillna(0) >= 36).astype(int)
    df["engagement_score"] = (
        df["n_products"].fillna(1)
        + df["has_savings_account"].fillna(0)
        + df["has_insurance"].fillna(0)
    )
    return df


def create_ratios_interactions(df, train_stats=None):
    """Ratios financiers et interactions cles.

    Parameters
    ----------
    df : DataFrame
    train_stats : dict or None
        Si fourni, contient exp_med et dti_med du train.
    """
    df = df.copy()
    df["asset_coverage"] = df["total_assets"].fillna(0) / (df["loan_amount"] + 1)
    if train_stats:
        exp_med = train_stats["exp_med"]
        dti_med = train_stats["dti_med"]
    else:
        exp_med = df["expenses_monthly"].median() if df["expenses_monthly"].notna().any() else 100000
        dti_med = df["debt_to_income"].median() if df["debt_to_income"].notna().any() else 0.3
    df["liquidity_ratio"] = (
        df["savings_balance"].fillna(0) + df["checking_avg_balance"].fillna(0)
    ) / (df["expenses_monthly"].fillna(exp_med) + 1)

    bureau_fill = df["bureau_score"].fillna(620)
    dti_fill = df["debt_to_income"].fillna(dti_med)
    df["bureau_x_dti"] = bureau_fill * dti_fill
    df["dpd_x_dti"] = df["dpd_severity"] * dti_fill
    return df


def create_log_transforms(df, train_stats=None):
    """Log-transform pour variables a forte asymetrie.

    Parameters
    ----------
    df : DataFrame
    train_stats : dict or None
        Si fourni, contient les medianes du train pour chaque colonne log.
        Les cles sont de la forme: {col}_med (ex: "income_monthly_med").
    """
    df = df.copy()
    for col in ["income_monthly", "loan_amount", "existing_debt", "total_assets"]:
        if train_stats and f"{col}_med" in train_stats:
            med = train_stats[f"{col}_med"]
        else:
            med = df[col].median() if df[col].notna().any() else 0
        df[f"log_{col}"] = np.log1p(df[col].fillna(med).clip(lower=0))
    return df


def create_demographic_bins(df):
    """Bins demographiques et ratios derives."""
    df = df.copy()
    df["age_young"] = (df["age"] < 30).astype(int)
    df["age_senior"] = (df["age"] > 55).astype(int)
    df["credit_age_ratio"] = df["avg_credit_age_months"].fillna(0) / (df["age"].fillna(30) * 12 + 1)
    return df


def create_risk_scores(df):
    """Score de risque composite."""
    df = df.copy()
    df["credit_risk_score"] = (
        (1 - df["bureau_score"].fillna(620) / 850) * 40
        + df["credit_utilization"].fillna(50) / 100 * 20
        + df["inquiries_6m"].fillna(0) * 3
        + df["derogatory_marks"].fillna(0) * 10
    )
    return df


def feature_engineering(df, train_stats=None):
    """Applique tout le feature engineering cible (26 nouvelles features).

    Chaque categorie est justifiee par l'EDA :
    - Indicateurs bureau : binning de la variable la plus predictive
    - Severite paiement : score composite des incidents
    - Capacite remboursement : derives du revenu net
    - Stabilite : anciennete, fidelite bancaire
    - Ratios/interactions : bureau x DTI, DPD x DTI
    - Log-transforms : pour variables a forte asymetrie
    - Scores composites : agregation de signaux faibles

    Parameters
    ----------
    df : DataFrame
    train_stats : dict or None
        Medianes et statistiques du train, necessaires en inference
        pour eviter que df.median() == valeur unique du client.
    """
    df = df.copy()
    df = create_bureau_indicators(df)
    df = create_payment_severity(df)
    df = create_repayment_capacity(df, train_stats=train_stats)
    df = create_stability_engagement(df)
    df = create_ratios_interactions(df, train_stats=train_stats)
    df = create_log_transforms(df, train_stats=train_stats)
    df = create_demographic_bins(df)
    df = create_risk_scores(df)
    return df
