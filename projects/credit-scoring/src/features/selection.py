"""
Feature Selection avancee par consensus multi-methodes.
Combine IV, correlation, LightGBM importance, RFE, et Boruta.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier

from ..config import (
    RANDOM_STATE, IV_THRESHOLDS, CORRELATION_THRESHOLD,
    VIF_THRESHOLD, N_FEATURES_MAX, N_FEATURES_MIN, CONSENSUS_VOTES
)


def select_by_iv(df_iv, df_iv_new=None, new_features=None, threshold=None):
    """Selection par Information Value (filtre univarie).

    Parameters
    ----------
    df_iv : DataFrame avec colonnes 'Variable', 'IV'
    df_iv_new : DataFrame IV des nouvelles features (optionnel)
    new_features : list des noms de nouvelles features
    threshold : float, seuil IV minimum (default: IV_THRESHOLDS["inutile"])

    Returns
    -------
    list de noms de features
    """
    if threshold is None:
        threshold = IV_THRESHOLDS["inutile"]

    feats = df_iv[df_iv["IV"] >= threshold]["Variable"].tolist()

    if df_iv_new is not None and new_features is not None:
        for v in new_features:
            mask = df_iv_new["Feature"] == v
            if mask.any() and df_iv_new.loc[mask, "IV"].values[0] >= threshold:
                if v not in feats:
                    feats.append(v)

    return feats


def select_by_correlation(df, features, iv_ref, threshold=None):
    """Supprime les features numeriques fortement correlees.

    Parameters
    ----------
    df : DataFrame
    features : list de features numeriques
    iv_ref : DataFrame avec colonnes 'Variable', 'IV'
    threshold : float, seuil de correlation

    Returns
    -------
    list de features conservees
    """
    if threshold is None:
        threshold = CORRELATION_THRESHOLD

    num_df = df[features].select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return features

    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold:
                iv_col = iv_ref[iv_ref["Variable"] == col]["IV"].values
                iv_idx = iv_ref[iv_ref["Variable"] == idx]["IV"].values
                if len(iv_col) > 0 and len(iv_idx) > 0:
                    to_drop.add(idx if iv_col[0] > iv_idx[0] else col)
                else:
                    to_drop.add(col)

    return [f for f in features if f not in to_drop]


def calc_vif_real(df_data, features):
    """Calcule le vrai VIF par regression OLS sur les donnees originales.

    VIF_j = 1 / (1 - R_j^2) ou R_j^2 est le R^2 de la regression
    de la feature j sur toutes les autres features.

    Parameters
    ----------
    df_data : DataFrame contenant les donnees originales
    features : list des noms de colonnes a evaluer

    Returns
    -------
    dict {feature: vif_value}
    """
    from sklearn.linear_model import LinearRegression

    X = df_data[features].dropna()
    vif = {}
    for i, col in enumerate(features):
        y = X.iloc[:, i].values
        X_others = X.drop(columns=[col]).values
        lr = LinearRegression()
        lr.fit(X_others, y)
        r2 = lr.score(X_others, y)
        vif[col] = 1.0 / (1.0 - min(r2, 0.9999))
    return vif


def select_by_vif(df_data, features, threshold=None):
    """Calcule le VIF reel par OLS et retire les features avec VIF > threshold.

    Parameters
    ----------
    df_data : DataFrame contenant les donnees originales (pas la matrice de correlation)
    features : list des noms de colonnes numeriques
    threshold : float, seuil VIF (default: VIF_THRESHOLD)

    Returns
    -------
    tuple (features_conservees, dict_vif)
    """
    if threshold is None:
        threshold = VIF_THRESHOLD

    num_features = [f for f in features if f in df_data.select_dtypes(include=[np.number]).columns]
    if len(num_features) < 2:
        return features, {}

    vif = calc_vif_real(df_data, num_features)

    high_vif = [k for k, v in vif.items() if v > threshold]
    return [f for f in features if f not in high_vif], vif


def select_by_lgb_importance(X, y, feature_names, top_k=35):
    """Selection par importance LightGBM.

    Returns
    -------
    top_features : list
    importances : DataFrame
    """
    lgb = LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    lgb.fit(X, y)

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": lgb.feature_importances_,
    }).sort_values("importance", ascending=False)

    top = imp.head(top_k)["feature"].tolist()
    return top, imp


def select_by_rfe(X, y, feature_names, n_features=35):
    """Recursive Feature Elimination avec LogReg."""
    lr = LogisticRegression(
        C=0.3, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    rfe = RFE(lr, n_features_to_select=min(n_features, X.shape[1]), step=5)
    rfe.fit(X, y)
    return [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]


def select_by_boruta(X, y, feature_names, top_k=35):
    """Boruta : test par permutation vs variables aleatoires."""
    try:
        from boruta import BorutaPy
    except ImportError:
        return select_by_lgb_importance(X, y, feature_names, top_k)[0]

    try:
        # Utiliser le top_k pour accelerer
        top_feats, imp = select_by_lgb_importance(X, y, feature_names, top_k)
        idx = [feature_names.index(f) for f in top_feats if f in feature_names]

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        )
        boruta = BorutaPy(rf, n_estimators="auto", perc=90, max_iter=50,
                          random_state=RANDOM_STATE, verbose=0)
        boruta.fit(X[:, idx], y.values.ravel())

        return [top_feats[i] for i in range(len(top_feats)) if boruta.support_[i]]
    except Exception as e:
        print(f"  Boruta fallback to LGB: {e}")
        return select_by_lgb_importance(X, y, feature_names, top_k)[0]


def consensus_selection(
    feats_iv, feats_corr, feats_lgb, feats_rfe, feats_boruta,
    feats_iv_cat, importance_df=None,
    n_min=N_FEATURES_MIN, n_max=N_FEATURES_MAX, votes_min=CONSENSUS_VOTES,
):
    """Selection par consensus : chaque feature recoit un vote par methode.

    Parameters
    ----------
    feats_iv, feats_corr, feats_lgb, feats_rfe, feats_boruta : lists
        Features selectionnees par chaque methode (numeriques)
    feats_iv_cat : list
        Categorielles avec IV > seuil
    importance_df : DataFrame avec colonnes 'feature', 'importance'
    n_min, n_max : int
        Bornes du nombre de features
    votes_min : int
        Nombre minimum de votes pour etre retenu

    Returns
    -------
    selected_num : list
    selected_cat : list
    votes_dict : dict
    """
    all_candidates = set(feats_iv + feats_corr + feats_lgb + feats_rfe + feats_boruta)
    votes = {}
    for f in all_candidates:
        v = 0
        if f in feats_iv: v += 1
        if f in feats_corr: v += 1
        if f in feats_lgb: v += 1
        if f in feats_rfe: v += 1
        if f in feats_boruta: v += 1
        votes[f] = v

    selected = sorted([f for f, v in votes.items() if v >= votes_min])

    # Ajuster si trop peu
    if len(selected) < n_min:
        extra = [f for f in feats_iv if f not in selected][:n_min - len(selected)]
        selected.extend(extra)

    # Ajuster si trop
    if len(selected) > n_max and importance_df is not None:
        imp_map = dict(zip(importance_df["feature"], importance_df["importance"]))
        selected.sort(key=lambda x: imp_map.get(x, 0), reverse=True)
        selected = selected[:n_max]

    return selected, feats_iv_cat, votes
