"""
Integration des variables macroeconomiques — Exigence IFRS 9.

IFRS 9 exige des scenarios forward-looking. Les provisions ECL
doivent incorporer l'impact des conditions macroeconomiques futures
sur la probabilite de defaut.

Ce module :
1. Merge les donnees macro avec les demandes de credit (par trimestre)
2. Cree des features macro interactives (ex: DTI × inflation)
3. Permet la simulation de scenarios (baseline, adverse, severe)
"""
import numpy as np
import pandas as pd


def load_macro_data(path="data/external/macro_uemoa.csv"):
    """Charge les donnees macroeconomiques UEMOA.

    Returns
    -------
    DataFrame avec colonnes : date, gdp_growth, inflation, unemployment, interbank_rate
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def merge_macro_to_applications(df_applications, df_macro):
    """Merge les variables macro par trimestre.

    Chaque demande de credit recoit les donnees macro du trimestre
    correspondant a sa date de demande.

    Parameters
    ----------
    df_applications : DataFrame avec 'application_date'
    df_macro : DataFrame avec 'date' (fin de mois)

    Returns
    -------
    DataFrame avec colonnes macro ajoutees
    """
    df = df_applications.copy()

    # Creer la colonne trimestre
    df["_quarter"] = df["application_date"].dt.to_period("Q")
    df_macro = df_macro.copy()
    df_macro["_quarter"] = df_macro["date"].dt.to_period("Q")
    df_macro = df_macro.drop(columns=["date"])

    # Merge
    df = df.merge(df_macro, on="_quarter", how="left")
    df = df.drop(columns=["_quarter"])

    return df


def create_macro_features(df):
    """Cree des features interactives basees sur les variables macro.

    L'impact du chomage ou de l'inflation depend du profil du client :
    - Un client avec DTI eleve est plus sensible a l'inflation
    - Un travailleur informel est plus sensible au chomage
    - Un credit long terme est plus sensible au taux directeur
    """
    df = df.copy()

    # Interaction DTI × inflation (le surendettement est aggrave par l'inflation)
    df["dti_x_inflation"] = (
        df["debt_to_income"].fillna(df["debt_to_income"].median())
        * df["inflation"].fillna(2.5)
    )

    # Interaction emploi informel × chomage
    df["informal_x_unemployment"] = (
        (df["employment_type"] == "Informal").astype(float)
        * df["unemployment"].fillna(6.0)
    )

    # Interaction duree pret × taux interbancaire
    df["term_x_interbank"] = (
        df["term_months"] / 12
        * df["interbank_rate"].fillna(4.0) / 100
    )

    # Ratio service dette / croissance PIB
    gdp = df["gdp_growth"].fillna(3.0).clip(lower=0.1)
    df["burden_to_gdp"] = (
        df["total_burden"] / gdp * 100
    ) if "total_burden" in df.columns else (df["payment_to_income"].fillna(0.3) / gdp * 100)

    # Score de stress macro composite
    df["macro_stress_score"] = (
        (df["inflation"].fillna(2.5) - 2.5) * 2
        + (df["unemployment"].fillna(6.0) - 6.0) * 1.5
        - (df["gdp_growth"].fillna(3.0) - 3.0) * 2
        + (df["interbank_rate"].fillna(4.0) - 4.0) * 1.0
    )

    return df


def generate_scenarios(df, scenario="baseline"):
    """Applique un scenario macroeconomique aux donnees.

    IFRS 9 exige 3 scenarios ponderes (ponderation standard BCEAO) :
    - Baseline (60%) : conditions normales, tendance actuelle
    - Adverse (30%) : recession moderee (crise regionale type BCEAO)
    - Severe (10%) : choc systemique (crise majeure, rare)

    Les chocs sont calibres sur les donnees BCEAO 2008-2020 :
    - Recession 2008-2009 : GDP -1.5pp a -2.0pp, inflation +2pp
    - Choc petrolier 2014-2015 : inflation +3pp, taux +2pp
    - COVID-19 2020 : GDP -3.5pp, unemployment +3pp
    Ref: BCEAO, "Rapport sur la Politique Monetaire", 2020.

    Parameters
    ----------
    df : DataFrame avec colonnes macro
    scenario : str
        "baseline", "adverse", ou "severe"

    Returns
    -------
    DataFrame avec variables macro ajustees
    """
    df_scenario = df.copy()

    shocks = {
        "baseline": {
            "gdp_growth": 0.0, "inflation": 0.0,
            "unemployment": 0.0, "interbank_rate": 0.0,
        },
        "adverse": {
            "gdp_growth": -1.5, "inflation": +2.0,
            "unemployment": +1.5, "interbank_rate": +1.5,
        },
        "severe": {
            "gdp_growth": -3.5, "inflation": +4.0,
            "unemployment": +3.0, "interbank_rate": +3.0,
        },
    }

    shock = shocks.get(scenario, shocks["baseline"])

    for var, delta in shock.items():
        if var in df_scenario.columns:
            df_scenario[var] = df_scenario[var].fillna(0) + delta

    # Recree les features interactives
    df_scenario = create_macro_features(df_scenario)

    return df_scenario


def scenario_weighted_pd(df, model, preprocessor, features, scenarios=None):
    """Calcule la PD ponderee par les scenarios IFRS 9.

    PD_finale = 0.60 × PD_baseline + 0.30 × PD_adverse + 0.10 × PD_severe

    Ponderations standard BCEAO (ref: BCEAO, "Rapport sur la Politique Monetaire", 2020).

    Parameters
    ----------
    df : DataFrame avec donnees macro
    model : modele fitted
    preprocessor : ColumnTransformer
    features : list
    scenarios : dict or None

    Returns
    -------
    DataFrame avec PD par scenario et PD ponderee
    """
    if scenarios is None:
        scenarios = {
            "baseline": 0.60,  # Poids standard BCEAO
            "adverse": 0.30,
            "severe": 0.10,
        }

    results = pd.DataFrame(index=df.index)

    for scenario, weight in scenarios.items():
        df_sc = generate_scenarios(df, scenario)
        X = df_sc[features]
        X_proc = preprocessor.transform(X)

        if isinstance(model, type) and "ExplainableBoosting" in str(model):
            X_proc_df = pd.DataFrame(X_proc, columns=features)
            pd_sc = model.predict_proba(X_proc_df)[:, 1]
        else:
            pd_sc = model.predict_proba(X_proc)[:, 1]

        results[f"pd_{scenario}"] = pd_sc
        results[f"pd_{scenario}_weighted"] = pd_sc * weight

    results["pd_weighted"] = results[[f"pd_{s}_weighted" for s in scenarios]].sum(axis=1)
    return results
