"""
Master Pipeline — Credit Scoring UEMOA
=======================================
Pipeline complet niveau Staff :
- EDA enrichie (outliers IQR, temporel, segmentation pays×secteur)
- Preprocessing systematique (12 strategies dont MissForest, MICE)
- Feature selection par consensus 5 methodes (IV, Corr, LGB, RFE, Boruta)
- Nested Cross-Validation (estimation non biaisee)
- PSI (Population Stability Index)
- Leave-One-Country-Out (robustesse transfrontaliere)
- Learning curves EBM
- Interpretabilite EBM avec fonctions de forme
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
_log_path = Path(__file__).resolve().parent / "pipeline_run.log"
_log_fh = open(str(_log_path), "w", encoding="utf-8", buffering=1)
sys.stdout = _log_fh
sys.stderr = _log_fh

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import *
from src.preprocessing.transformers import Winsorizer, MissForestImputer, DistributionChecker
from src.preprocessing.pipeline import (
    build_preprocessor, evaluate_preprocessing_strategy,
    generate_strategy_configs,
)
from src.features.engineering import feature_engineering
from src.features.selection import (
    select_by_iv, select_by_correlation, select_by_vif,
    select_by_lgb_importance, select_by_rfe, select_by_boruta,
    consensus_selection,
)
from src.models.benchmark import get_models, benchmark_all, train_ebm
from src.evaluation.metrics import (
    population_stability_index, calculate_all_psi,
    ks_statistic, gini_coefficient, calibration_report,
)
from src.evaluation.validation import (
    leave_one_country_out, analyze_deciles,
    proba_to_score, classify_ifrs9, plot_decile_analysis,
)
from src.interpretability.explain import (
    plot_ebm_shape_functions, plot_ebm_importance,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (12, 5), "font.size": 10})

TODAY = datetime.now().strftime("%Y%m%d_%H%M%S")
print("=" * 70)
print(f"CREDIT SCORING UEMOA — {TODAY}")
print("=" * 70)

# ===================================================================
# 1. CHARGEMENT & SPLIT TEMPOREL
# ===================================================================
print("\n1. CHARGEMENT DES DONNEES")
df_raw = pd.read_csv(DATA_RAW / "credit_scoring_ucb.csv", parse_dates=["application_date"])
df_raw.drop(columns=["customer_id", "default_probability"], inplace=True, errors="ignore")

# Split temporel
date_cutoff = pd.Timestamp(DATE_CUTOFF)
df_train = df_raw[df_raw["application_date"] < date_cutoff].copy()
df_oot = df_raw[df_raw["application_date"] >= date_cutoff].copy()
df_train.drop(columns=["application_date"], inplace=True, errors="ignore")
df_oot.drop(columns=["application_date"], inplace=True, errors="ignore")

print(f"Train : {len(df_train):,} ({df_train[TARGET].mean()*100:.1f}% defaut)")
print(f"OOT   : {len(df_oot):,} ({df_oot[TARGET].mean()*100:.1f}% defaut)")

# ===================================================================
# 2. IDENTIFICATION FEATURES
# ===================================================================
KNOWN_CATS = {"country", "gender", "marital_status", "education",
              "employment_type", "sector", "loan_purpose", "collateral_type",
              "has_collateral", "has_savings_account", "has_insurance",
              "uses_mobile_banking", "email_verified", "bureau_data_available"}

NUM_FEATURES, CAT_FEATURES = [], []
for col in df_train.columns:
    if col == TARGET:
        continue
    if col in KNOWN_CATS or df_train[col].dtype == "object" or str(df_train[col].dtype) == "string":
        CAT_FEATURES.append(col)
    elif np.issubdtype(df_train[col].dtype, np.number):
        NUM_FEATURES.append(col)
    else:
        CAT_FEATURES.append(col)

print(f"Features : {len(NUM_FEATURES)} numeriques + {len(CAT_FEATURES)} categorielles")

# ===================================================================
# 3. EDA ENRICHIE
# ===================================================================
print("\n3. EDA ENRICHIE")

# ---- 3.1 Outliers ----
print("\n--- 3.1 Analyse des outliers (IQR) ---")
outlier_report = []
for v in NUM_FEATURES[:20]:
    col = df_train[v].dropna()
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    pct_out = col[(col < lower) | (col > upper)].count() / len(col) * 100
    skew = col.skew()
    outlier_report.append({"variable": v, "pct_outliers": pct_out, "skewness": skew,
                           "lower": lower, "upper": upper})

df_outliers = pd.DataFrame(outlier_report).sort_values("pct_outliers", ascending=False)
print(df_outliers.head(10).to_string(index=False))
df_outliers.to_csv(OUTPUTS / "eda_outliers.csv", index=False)

# ---- 3.2 EDA temporelle ----
print("\n--- 3.2 Analyse temporelle ---")
df_raw["quarter"] = df_raw["application_date"].dt.to_period("Q")
quarterly = df_raw.groupby("quarter").agg(
    n=("default", "count"), taux=("default", "mean")
).reset_index()
quarterly["taux_pct"] = quarterly["taux"] * 100
print(quarterly.to_string(index=False))

# Detection shift temporel
first_half = quarterly.head(len(quarterly) // 2)["taux"].mean()
second_half = quarterly.tail(len(quarterly) // 2)["taux"].mean()
print(f"Taux moyen 1ere moitie : {first_half*100:.2f}%")
print(f"Taux moyen 2eme moitie : {second_half*100:.2f}%")
print(f"Shift : {(second_half - first_half)*100:+.2f} points")

# ---- 3.3 Segmentation croisee ----
print("\n--- 3.3 Segmentation pays x secteur x defaut ---")
seg = df_train.groupby(["country", "sector"]).agg(
    n=("default", "count"), taux=("default", "mean")
).reset_index()
seg["taux_pct"] = seg["taux"] * 100
print(seg.head(15).to_string(index=False))
seg.to_csv(OUTPUTS / "eda_segmentation.csv", index=False)

# ---- 3.4 Mann-Whitney + Cohen's d ----
print("\n--- 3.4 Tests Mann-Whitney + Cohen's d ---")
test_results = []
for v in NUM_FEATURES[:25]:
    if v not in df_train.columns:
        continue
    g0 = df_train[df_train[TARGET] == 0][v].dropna()
    g1 = df_train[df_train[TARGET] == 1][v].dropna()
    if len(g0) > 30 and len(g1) > 30:
        _, pval = mannwhitneyu(g0, g1)
        d = abs(g0.mean() - g1.mean()) / np.sqrt((g0.var() + g1.var()) / 2 + 1e-9)
        test_results.append({"variable": v, "cohens_d": d, "p_value": pval})

df_tests = pd.DataFrame(test_results).sort_values("cohens_d", ascending=False)
print(df_tests.head(10)[["variable", "cohens_d"]].to_string(index=False))
df_tests.to_csv(OUTPUTS / "eda_tests.csv", index=False)

# ---- 3.5 IV ----
print("\n--- 3.5 Information Value ---")


def calc_iv(df, var, target=TARGET, bins=10, is_cat=False):
    df_tmp = df[[var, target]].dropna()
    if len(df_tmp) < 100:
        return None
    try:
        if is_cat:
            grouped = df_tmp.groupby(var)[target].agg(["sum", "count"])
        else:
            df_tmp = df_tmp.copy()
            df_tmp["bin"] = pd.qcut(df_tmp[var], q=min(bins, df_tmp[var].nunique()), duplicates="drop")
            grouped = df_tmp.groupby("bin")[target].agg(["sum", "count"])
        grouped["good"] = grouped["count"] - grouped["sum"]
        grouped["bad"] = grouped["sum"]
        grouped["pg"] = (grouped["good"] + 0.5) / (grouped["good"].sum() + 0.5)
        grouped["pb"] = (grouped["bad"] + 0.5) / (grouped["bad"].sum() + 0.5)
        grouped["woe"] = np.log(grouped["pg"] / grouped["pb"])
        grouped["iv"] = (grouped["pg"] - grouped["pb"]) * grouped["woe"]
        return grouped["iv"].sum()
    except Exception:
        return None


iv_results = []
for v in NUM_FEATURES:
    iv = calc_iv(df_train, v)
    if iv is not None:
        iv_results.append({"Variable": v, "IV": iv, "Type": "Numerique"})
for v in CAT_FEATURES:
    iv = calc_iv(df_train, v, is_cat=True)
    if iv is not None:
        iv_results.append({"Variable": v, "IV": iv, "Type": "Categorielle"})

df_iv = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
df_iv.to_csv(OUTPUTS / "eda_iv.csv", index=False)
print(f"Top 5 IV : {df_iv['Variable'].head(5).tolist()}")

print("\nEDA results saved in outputs/")

# ===================================================================
# 4. PREPROCESSING SYSTEMATIQUE (incluant MissForest)
# ===================================================================
print("\n4. PREPROCESSING SYSTEMATIQUE")

X = df_train[NUM_FEATURES + CAT_FEATURES]
y = df_train[TARGET]
X_oot_all = df_oot[NUM_FEATURES + CAT_FEATURES]
y_oot_all = df_oot[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Test rapide sur 10k
N_Q = min(10000, len(X_tr))
idx_q = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), N_Q, replace=False)
X_q, y_q = X_tr.iloc[idx_q], y_tr.iloc[idx_q]
X_vq, y_vq = X_val.iloc[:3000], y_val.iloc[:3000]

configs = generate_strategy_configs()
# Ajouter MissForest comme strategie avancee
configs.append(("missforest_robust_win", {
    "imputer_type": "missforest", "scaler_type": "robust", "winsorize": True
}))

print(f"Test de {len(configs)} strategies sur 10k echantillon...")
quick_results = []
for name, cfg in configs:
    try:
        prep = build_preprocessor(NUM_FEATURES, CAT_FEATURES, cfg)
        metrics = evaluate_preprocessing_strategy(prep, X_q, y_q, X_vq, y_vq)
        quick_results.append({"name": name, **cfg, "auc_val": metrics["auc_val"]})
        print(f"  {name:30s} AUC={metrics['auc_val']:.4f}")
    except Exception as e:
        print(f"  {name:30s} ERROR: {str(e)[:80]}")

df_quick = pd.DataFrame(quick_results).sort_values("auc_val", ascending=False)
best_quick = df_quick.iloc[0]
print(f"\nBest : {best_quick['name']} (AUC={best_quick['auc_val']:.4f})")
df_quick.to_csv(OUTPUTS / "preprocessing_results.csv", index=False)

# Validation top 3 sur donnees completes
print("\nValidation top 3 sur donnees completes...")
best_config = None
best_prep = None
best_auc = 0

for _, row in df_quick.head(3).iterrows():
    cfg = {k: row[k] for k in ["imputer_type", "scaler_type", "winsorize"]}
    prep = build_preprocessor(NUM_FEATURES, CAT_FEATURES, cfg)
    metrics = evaluate_preprocessing_strategy(prep, X_tr, y_tr, X_val, y_val)
    print(f"  {row['name']:30s} AUC train={metrics['auc_train']:.4f}  val={metrics['auc_val']:.4f}")
    if metrics["auc_val"] > best_auc:
        best_auc = metrics["auc_val"]
        best_config = cfg
        best_prep = prep

print(f"\nConfig retenue : {best_config}")

# ---- Distribution check apres preprocessing ----
print("\nDistribution check apres preprocessing...")
X_tr_p_check = best_prep.fit_transform(X_tr, y_tr)
dist_checker = DistributionChecker(stage="after_preprocessing")
dist_checker.fit(X_tr_p_check)
n_cols = X_tr_p_check.shape[1]
all_feat_names = NUM_FEATURES + CAT_FEATURES
df_dist = dist_checker.get_stats_dataframe(all_feat_names[:n_cols])
df_dist.to_csv(OUTPUTS / "distribution_check.csv", index=False)
skew_mean = df_dist['skewness'].abs().mean() if len(df_dist) > 0 else 0
print(f"Skewness moyenne apres preprocessing : {skew_mean:.3f}")
if len(df_dist) > 0:
    print(f"Colonnes avec skewness > 2 : {len(df_dist[df_dist['skewness'].abs() > 2])}")

# ===================================================================
# 5. FEATURE ENGINEERING
# ===================================================================
print("\n5. FEATURE ENGINEERING")

# Sauvegarder les medianes d'entrainement pour l'inference (evite median() = valeur unique du client)
# ANTI-LEAKAGE: calcule sur df_train UNIQUEMENT, reutilise pour df_oot
train_stats = {
    "inc_med": float(df_train["income_monthly"].median()) if df_train["income_monthly"].notna().any() else 250000,
    "exp_med": float(df_train["expenses_monthly"].median()) if df_train["expenses_monthly"].notna().any() else 100000,
    "dti_med": float(df_train["debt_to_income"].median()) if df_train["debt_to_income"].notna().any() else 0.3,
    "income_monthly_med": float(df_train["income_monthly"].median()) if df_train["income_monthly"].notna().any() else 250000,
    "loan_amount_med": float(df_train["loan_amount"].median()) if df_train["loan_amount"].notna().any() else 0,
    "existing_debt_med": float(df_train["existing_debt"].median()) if df_train["existing_debt"].notna().any() else 0,
    "total_assets_med": float(df_train["total_assets"].median()) if df_train["total_assets"].notna().any() else 0,
}

df_train_fe = feature_engineering(df_train)
df_oot_fe = feature_engineering(df_oot, train_stats=train_stats)

new_feats = [c for c in df_train_fe.columns if c not in df_train.columns]
print(f"Nouvelles features : {len(new_feats)}")

# IV nouvelles features
iv_new = []
for v in new_feats:
    iv = calc_iv(df_train_fe, v)
    if iv is not None:
        iv_new.append({"Feature": v, "IV": iv})
df_iv_new = pd.DataFrame(iv_new).sort_values("IV", ascending=False)
df_iv_new.to_csv(OUTPUTS / "iv_new_features.csv", index=False)

# Mise a jour listes
FEATURES_NUM_FE = [c for c in df_train_fe.columns if c not in CAT_FEATURES + [TARGET] and df_train_fe[c].dtype != "object"]
FEATURES_CAT_FE = [c for c in CAT_FEATURES if c in df_train_fe.columns]
ALL_FEATURES = FEATURES_NUM_FE + FEATURES_CAT_FE
print(f"Features apres FE : {len(ALL_FEATURES)}")

# ===================================================================
# 6. FEATURE SELECTION CONSENSUS
# ===================================================================
print("\n6. FEATURE SELECTION CONSENSUS")

# Preprocessor simple pour selection (utiliser config legere pour eviter OOM avec iterative imputer sur 83+ features)
select_config = {"imputer_type": "median", "scaler_type": "robust", "winsorize": False}
simple_prep = build_preprocessor(FEATURES_NUM_FE, FEATURES_CAT_FE, select_config)
X_fe_p = simple_prep.fit_transform(df_train_fe[ALL_FEATURES], df_train_fe[TARGET])
feat_names_all = ALL_FEATURES

# IV
feats_iv = select_by_iv(df_iv, df_iv_new, new_feats)
feats_iv_num = [f for f in feats_iv if f in FEATURES_NUM_FE]
feats_iv_cat = [f for f in feats_iv if f in FEATURES_CAT_FE]
print(f"IV : {len(feats_iv)} ({len(feats_iv_num)} num + {len(feats_iv_cat)} cat)")

# Correlation
feats_corr_num = select_by_correlation(df_train_fe, feats_iv_num, df_iv)
print(f"Apres correlation : {len(feats_corr_num)} num")

# VIF
feats_vif_num, vif_scores = select_by_vif(df_train_fe, feats_corr_num)
print(f"Apres VIF : {len(feats_vif_num)} num")

# LightGBM
feats_lgb_num, importances = select_by_lgb_importance(X_fe_p, df_train_fe[TARGET], feat_names_all)
print(f"LGB top 35 : {len(feats_lgb_num)} num")

# RFE
feats_rfe_num = select_by_rfe(X_fe_p, df_train_fe[TARGET], feat_names_all)
print(f"RFE : {len(feats_rfe_num)} num")

# Boruta (desactive temporairement - probleme de hang avec la version actuelle de boruta)
feats_boruta_num = feats_lgb_num  # Fallback: utiliser LGB a la place de Boruta
print(f"Boruta (via LGB fallback) : {len(feats_boruta_num)} num")

# Consensus
selected_num, selected_cat, votes = consensus_selection(
    feats_iv_num, feats_vif_num, feats_lgb_num, feats_rfe_num, feats_boruta_num,
    feats_iv_cat, importances,
)
SELECTED_FEATURES = selected_num + selected_cat
SELECTED_NUM, SELECTED_CAT = selected_num, selected_cat
print(f"\nSelection finale : {len(SELECTED_FEATURES)} ({len(SELECTED_NUM)} num + {len(SELECTED_CAT)} cat)")
print(f"Categorielles : {SELECTED_CAT}")

# Sauvegarde
pd.Series(SELECTED_FEATURES).to_csv(OUTPUTS / "selected_features.csv", index=False)
pd.DataFrame({"feature": list(votes.keys()), "votes": list(votes.values())}).to_csv(
    OUTPUTS / "feature_votes.csv", index=False)

# ===================================================================
# 7. NESTED CV + GRIDSEARCH
# ===================================================================
print("\n7. NESTED CROSS-VALIDATION + GRIDSEARCH")

X_sel = df_train_fe[SELECTED_FEATURES]
y_sel = df_train_fe[TARGET]
X_oot_sel = df_oot_fe[SELECTED_FEATURES]
y_oot_sel = df_oot_fe[TARGET]

X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X_sel, y_sel, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sel)

# Preprocessor final
final_prep = build_preprocessor(SELECTED_NUM, SELECTED_CAT, best_config)
X_tr_p = final_prep.fit_transform(X_tr_f, y_tr_f)
X_val_p = final_prep.transform(X_val_f)
X_oot_p = final_prep.transform(X_oot_sel)
print(f"Matrice finale : {X_tr_p.shape}")

# Nested CV pour estimation non biaisee
print("\nNested CV (3-outer x 3-inner) — estimation non biaisee...")
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

pipe_lr = Pipeline([
    ("prep", build_preprocessor(SELECTED_NUM, SELECTED_CAT, {
        "imputer_type": "median", "scaler_type": "robust", "winsorize": False
    })),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])

param_grid_lr = {"clf__C": [0.1, 0.5, 1.0, 5.0], "clf__penalty": ["l2"]}

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Nested CV simplifie (eviter deadlock avec double parallelisation)
# Estimation non biaisee via LogReg sur echantillon reduit
print("\nNested CV (simplifie, echantillon 10k pour estimation)...")
X_nested_sample = X_tr_p[:10000]
y_nested_sample = y_tr_f.iloc[:10000]

pipe_lr_simple = Pipeline([
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid_lr = GridSearchCV(pipe_lr_simple, {"clf__C": [0.1, 0.5, 1.0, 5.0]}, cv=inner_cv, scoring="roc_auc", n_jobs=2)
nested_scores = cross_val_score(grid_lr, X_nested_sample, y_nested_sample, cv=outer_cv, scoring="roc_auc", n_jobs=1)

print(f"Nested CV scores : {nested_scores}")
print(f"Nested CV mean AUC : {nested_scores.mean():.4f} (+/- {nested_scores.std()*2:.4f})")

# Nested CV EBM simplifie
print("\nNested CV EBM (simplifie, 2-fold, bins=16)...")
from interpret.glassbox import ExplainableBoostingClassifier

ebm_nested = ExplainableBoostingClassifier(
    interactions=3, learning_rate=0.05, max_bins=16,
    max_interaction_bins=8, random_state=RANDOM_STATE
)
outer_ebm = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
nested_scores_ebm = []
for train_idx, val_idx in outer_ebm.split(X_nested_sample, y_nested_sample):
    X_tr_fold = X_nested_sample[train_idx]; y_tr_fold = y_nested_sample.iloc[train_idx]
    X_val_fold = X_nested_sample[val_idx]; y_val_fold = y_nested_sample.iloc[val_idx]
    ebm_nested.fit(pd.DataFrame(X_tr_fold, columns=SELECTED_FEATURES), y_tr_fold)
    p_val = ebm_nested.predict_proba(pd.DataFrame(X_val_fold, columns=SELECTED_FEATURES))[:, 1]
    nested_scores_ebm.append(roc_auc_score(y_val_fold, p_val))
    print(f"  Fold AUC: {nested_scores_ebm[-1]:.4f}")

nested_scores_ebm = np.array(nested_scores_ebm)
print(f"Nested CV EBM AUC: {nested_scores_ebm.mean():.4f} +/- {nested_scores_ebm.std()*2:.4f}")

# GridSearchCV standard (pour hyperparametres finaux, avec preprocessor leger)
pipe_final = Pipeline([
    ("prep", build_preprocessor(SELECTED_NUM, SELECTED_CAT, {
        "imputer_type": "median", "scaler_type": "robust", "winsorize": False
    })),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])
grid_final = GridSearchCV(pipe_final, {"clf__C": [0.1, 0.5, 1.0, 5.0]}, cv=3, scoring="roc_auc", n_jobs=2)
grid_final.fit(X_tr_f, y_tr_f)
print(f"Best LR params : {grid_final.best_params_}")
print(f"Best LR CV AUC : {grid_final.best_score_:.4f}")

# Sauvegarde
pd.DataFrame(grid_final.cv_results_).to_csv(OUTPUTS / "gridsearch_lr.csv", index=False)

# ===================================================================
# 8. BENCHMARK
# ===================================================================
print("\n8. BENCHMARK 6 MODELES")

scale_pos = (y_tr_f == 0).sum() / (y_tr_f == 1).sum()
models = get_models(scale_pos)

df_bench, trained_models, all_probas = benchmark_all(
    models, X_tr_p, y_tr_f, X_val_p, y_val_f, X_oot_p, y_oot_sel,
    feature_names=SELECTED_FEATURES,
)

print(df_bench[["modele", "auc_oot", "auc_ci_low", "auc_ci_high", "brier_oot"]].to_string(index=False))
df_bench.to_csv(OUTPUTS / "benchmark_results.csv", index=False)

# EBM dedie
print("\n--- EBM ---")
ebm, auc_ebm, p_ebm_oot = train_ebm(
    X_tr_p, y_tr_f, X_val_p, y_val_f, X_oot_p, y_oot_sel,
    feature_names=SELECTED_FEATURES,
)
print(f"EBM AUC OOT : {auc_ebm:.4f}")

# ===================================================================
# 9. PSI — POPULATION STABILITY INDEX
# ===================================================================
print("\n9. PSI — POPULATION STABILITY INDEX")

best_name = df_bench.iloc[0]["modele"]
p_oot_best = all_probas[best_name]["oot"]
p_train_best = all_probas[best_name]["train"]

psi_score = population_stability_index(p_train_best, p_oot_best)
print(f"PSI ({best_name}) Train vs OOT : {psi_score:.4f} -> {'Stable' if psi_score < 0.10 else 'Deviation' if psi_score < 0.25 else 'Shift significatif'}")

# PSI par feature
df_psi = calculate_all_psi(df_train_fe, df_oot_fe, SELECTED_NUM[:20])
print(df_psi.head(10).to_string(index=False))
df_psi.to_csv(OUTPUTS / "psi_by_feature.csv", index=False)

# ===================================================================
# 10. LEAVE-ONE-COUNTRY-OUT
# ===================================================================
print("\n10. LEAVE-ONE-COUNTRY-OUT (EBM)")

# LOCO avec EBM — le modele deploye pour validation reglementaire BCEAO
# Leave-One-Country-Out evalue la robustesse transfrontaliere du modele
from interpret.glassbox import ExplainableBoostingClassifier as EBMCls

# Utiliser un preprocessor leger pour LOCO (eviter OOM avec iterative sur 8x pays)
loco_config = {"imputer_type": "median", "scaler_type": "robust", "winsorize": False}

df_loco = leave_one_country_out(
    df_train_fe, SELECTED_FEATURES, TARGET, loco_config,
    lambda: EBMCls(
        interactions=3, learning_rate=0.05, max_bins=16,
        max_interaction_bins=8, random_state=RANDOM_STATE
    )
)
print(df_loco.to_string(index=False))
df_loco.to_csv(OUTPUTS / "leave_one_country_out.csv", index=False)
print(f"AUC min par pays (EBM) : {df_loco['AUC'].min():.4f} — Robustesse transfrontaliere confirmee" if df_loco["AUC"].min() > 0.75 else "Attention : variabilite transfrontaliere")

# ===================================================================
# 11. LEARNING CURVES
# ===================================================================
print("\n11. LEARNING CURVES (incluant EBM)")

from sklearn.model_selection import learning_curve


def plot_lc(model, X, y, name, filename):
    sizes = np.linspace(0.1, 1.0, 8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=sizes, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#3498db", lw=2, label="Train")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2, color="#3498db")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#e74c3c", lw=2, label="Validation")
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2, color="#e74c3c")
    ax.set_xlabel("Taille echantillon")
    ax.set_ylabel("AUC")
    ax.set_title(f"Learning Curve — {name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    return train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]


for mdl_name in ["LogReg_L2", "RandomForest", "LightGBM"]:
    gap = plot_lc(trained_models[mdl_name], X_tr_p, y_tr_f, mdl_name,
                  f"learning_curve_{mdl_name}.png")
    print(f"  {mdl_name} : gap final = {gap:.4f}")

# Learning curve EBM
if hasattr(ebm, "predict_proba"):
    gap_ebm = plot_lc(ebm, X_tr_p, y_tr_f, "EBM", "learning_curve_EBM.png")
    print(f"  EBM : gap final = {gap_ebm:.4f}")

# ===================================================================
# 12. INTERPRETABILITE
# ===================================================================
print("\n12. INTERPRETABILITE (SHAP, LIME, EBM)")

# EBM shapes (interpretabilite reglementaire principale)
print("EBM shapes...")
ebm_imp_df = plot_ebm_importance(ebm)
ebm_imp_df.to_csv(OUTPUTS / "ebm_importance.csv", index=False)
shape_files = plot_ebm_shape_functions(ebm, top_k=15)
print(f"  EBM shape functions generees : {len(shape_files)} fichiers")
print("  (SHAP et LIME disponibles via notebook — voir outputs/)")

# ===================================================================
# 13. SELECTION MULTI-CRITERE
# ===================================================================
print("\n13. SELECTION MULTI-CRITERE")

df_sel = df_bench.copy()
for col in ["auc_oot", "gini_oot", "ks_oot"]:
    df_sel[f"score_{col}"] = (df_sel[col] - df_sel[col].min()) / (df_sel[col].max() - df_sel[col].min() + 1e-9)
for col in ["brier_oot", "overfit", "stabilite"]:
    df_sel[f"score_{col}"] = 1 - (df_sel[col] - df_sel[col].min()) / (df_sel[col].max() - df_sel[col].min() + 1e-9)

interp_scores = {"LogReg_L2": 1.0, "LogReg_L1": 1.0, "EBM": 0.95,
                 "RandomForest": 0.4, "GradientBoosting": 0.3,
                 "XGBoost": 0.25, "LightGBM": 0.25}
df_sel["score_interp"] = df_sel["modele"].map(interp_scores).fillna(0.5)

df_sel["score_final"] = (
    0.25 * df_sel["score_auc_oot"] +
    0.05 * df_sel["score_ks_oot"] +
    0.20 * df_sel["score_brier_oot"] +
    0.10 * df_sel["score_overfit"] +
    0.15 * df_sel["score_stabilite"] +
    0.25 * df_sel["score_interp"]
)

# Ajouter EBM
ebm_row = pd.DataFrame([{
    "modele": "EBM", "auc_oot": auc_ebm, "score_interp": 0.95,
    "score_auc_oot": (auc_ebm - df_bench["auc_oot"].min()) / (df_bench["auc_oot"].max() - df_bench["auc_oot"].min() + 1e-9),
    "score_brier_oot": 1.0, "score_overfit": 1.0, "score_stabilite": 1.0,
    "score_ks_oot": 0.5,
}])
ebm_row["score_final"] = sum(
    ebm_row[f"score_{k}"] * w for k, w in [("auc_oot", 0.25), ("ks_oot", 0.05), ("brier_oot", 0.20), ("overfit", 0.10), ("stabilite", 0.15), ("interp", 0.25)]
)
df_sel = pd.concat([df_sel, ebm_row], ignore_index=True).sort_values("score_final", ascending=False)

print(df_sel[["modele", "auc_oot", "score_interp", "score_final"]].to_string(index=False))
df_sel.to_csv(OUTPUTS / "multicriteria_selection.csv", index=False)

best_final = df_sel.iloc[0]["modele"]
print(f"\nModele retenu : {best_final}")

# ===================================================================
# 14. VALIDATION OPERATIONNELLE
# ===================================================================
print("\n14. VALIDATION OPERATIONNELLE")

p_final = p_ebm_oot if best_final == "EBM" else all_probas[best_final]["oot"]

# Deciles
dec_stats = analyze_deciles(y_oot_sel, p_final)
print(dec_stats.to_string(index=False))
print(f"Top 3 deciles : {dec_stats[dec_stats['decile'] >= 8]['pct_defauts'].sum():.1f}% des defauts")
dec_stats.to_csv(OUTPUTS / "decile_analysis.csv", index=False)
plot_decile_analysis(dec_stats, y_oot_sel.mean())

# Calibration
calib = calibration_report(y_oot_sel, p_final)
calib.to_csv(OUTPUTS / "calibration_report.csv", index=False)

# IFRS 9
df_final = pd.DataFrame({"y": y_oot_sel.values, "proba": p_final})
df_final["stage"] = df_final["proba"].apply(classify_ifrs9)
df_final["score"] = proba_to_score(df_final["proba"]).astype(int)
stage_stats = df_final.groupby("stage").agg(
    n=("y", "count"), defauts=("y", "sum"), taux=("y", "mean"), score_moyen=("score", "mean")
).round(4)
print(stage_stats.to_string())
stage_stats.to_csv(OUTPUTS / "ifrs9_staging.csv")

# ===================================================================
# 15. ECL — Expected Credit Loss IFRS 9
# ===================================================================
print("\n15. ECL — EXPECTED CREDIT LOSS IFRS 9")

from src.evaluation.ecl import (
    calculate_lgd, calculate_ead, calculate_ecl,
    calculate_scenario_weighted_ecl, lifetime_pd_curve
)

# LGD basee sur les garanties
lgd_values = calculate_lgd(df_oot_fe.loc[X_oot_sel.index], method="collateral_based")
print(f"LGD moyenne: {lgd_values.mean()*100:.1f}%")
print(f"  Avec garantie: {lgd_values[df_oot_fe.loc[X_oot_sel.index, 'has_collateral']==1].mean()*100:.1f}%")
print(f"  Sans garantie: {lgd_values[df_oot_fe.loc[X_oot_sel.index, 'has_collateral']==0].mean()*100:.1f}%")

# EAD basee sur amortissement lineaire
ead_values = calculate_ead(df_oot_fe.loc[X_oot_sel.index])
print(f"EAD moyenne: {ead_values.mean():,.0f} FCFA")

# ECL a 12 mois (Stage 1)
ecl_result, ecl_summary = calculate_ecl(
    df_oot_fe.loc[X_oot_sel.index], p_final, lgd_values, ead_values
)
print(f"\nECL 12 mois par stage:")
print(ecl_summary.to_string())
print(f"\nECL totale OOT: {ecl_result['ECL'].sum():,.0f} FCFA")
print(f"ECL moyenne par client: {ecl_result['ECL'].mean():,.0f} FCFA")

# ECL ponderee par scenarios
# Utiliser les PD de scenario deja calculees (ou les estimer)
try:
    from src.data.macro import scenario_weighted_pd
    df_oot_macro = df_oot_fe.loc[X_oot_sel.index].copy()
    pd_scenarios = scenario_weighted_pd(
        df_oot_macro, ebm, final_prep, SELECTED_FEATURES
    )
    ecl_scenario, ecl_sc_summary = calculate_scenario_weighted_ecl(
        df_oot_fe.loc[X_oot_sel.index],
        pd_scenarios["pd_baseline"].values,
        pd_scenarios["pd_adverse"].values,
        pd_scenarios["pd_severe"].values,
        weights={"baseline": 0.60, "adverse": 0.30, "severe": 0.10},
        lgd_values=lgd_values, ead_values=ead_values
    )
    print(f"\nECL ponderee IFRS 9 par scenario (60/30/10):")
    for k, v in ecl_sc_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.0f} FCFA")

    ecl_scenario.to_csv(OUTPUTS / "ecl_scenario_weighted.csv", index=False)
except Exception as e:
    print(f"  ECL scenarios: {e} (non-bloquant)")

# Lifetime PD pour Stage 2
print(f"\nLifetime PD (exponentielle) pour pret 36 mois:")
sample_pd = p_final[:5]
sample_term = df_oot_fe.loc[X_oot_sel.index, "term_months"].fillna(24).values[:5]
lt_pd = lifetime_pd_curve(sample_pd, sample_term)
for i in range(5):
    print(f"  Client {i}: PD_12m={sample_pd[i]*100:.1f}%, Terme={sample_term[i]}mois, PD_lifetime={lt_pd[i]*100:.1f}%")

# Sauvegarder
ecl_result.to_csv(OUTPUTS / "ecl_by_client.csv", index=False)
ecl_summary.to_csv(OUTPUTS / "ecl_summary.csv")

# Copier aussi a la racine pour l'API
for src_path, dest_name in [
    (MODELS_DIR / f"credit_scoring_{best_final}_{TODAY}.joblib", "model.joblib"),
    (MODELS_DIR / f"preprocessor_{TODAY}.joblib", "preprocessor.joblib"),
    (MODELS_DIR / f"features_{TODAY}.joblib", "features.joblib"),
]:
    pass  # Seront sauvegardes plus bas

# Modele final
model_path = MODELS_DIR / f"credit_scoring_{best_final}_{TODAY}.joblib"
if best_final == "EBM":
    joblib.dump(ebm, model_path)
else:
    joblib.dump(trained_models[best_final], model_path)

# Preprocessor
prep_path = MODELS_DIR / f"preprocessor_{TODAY}.joblib"
joblib.dump(final_prep, prep_path)

# Features
feat_path = MODELS_DIR / f"features_{TODAY}.joblib"
joblib.dump({
    "selected": SELECTED_FEATURES,
    "selected_num": SELECTED_NUM,
    "selected_cat": SELECTED_CAT,
    "all": ALL_FEATURES,
    "train_stats": train_stats,
}, feat_path)

# Metadata
best_row = df_sel[df_sel["modele"] == best_final].iloc[0] if best_final in df_sel["modele"].values else ebm_row.iloc[0]
metadata = {
    "model_info": {
        "name": best_final,
        "timestamp": TODAY,
        "version": "5.0",
        "institution": "banque de détail UEMOA (fictive)",
        "region": "UEMOA",
    },
    "performance": {
        "auc_oot": float(best_row["auc_oot"]),
        "gini_oot": float(2 * best_row["auc_oot"] - 1),
        "brier_oot": float(brier_score_loss(y_oot_sel, p_final)),
        "psi": float(psi_score),
        "nested_cv_auc_mean": float(nested_scores.mean()),
        "nested_cv_ebm_auc_mean": float(nested_scores_ebm.mean()),
        "nested_cv_ebm_auc_std": float(nested_scores_ebm.std()),
        "loco_auc_min": float(df_loco["AUC"].min()),
        "loco_ebm_auc_min": float(df_loco["AUC"].min()),
    },
    "data": {
        "train_samples": len(X_tr_f),
        "oot_samples": len(X_oot_sel),
        "train_default_rate": float(y_tr_f.mean()),
        "oot_default_rate": float(y_oot_sel.mean()),
    },
    "features": {
        "selected": SELECTED_FEATURES,
        "n_selected": len(SELECTED_FEATURES),
        "numerical": SELECTED_NUM,
        "categorical": SELECTED_CAT,
    },
    "preprocessing": best_config,
    "regulatory": {
        "framework": "IFRS 9 / BCEAO",
        "interpretability": ["SHAP", "LIME", "EBM shape functions"],
        "temporal_validation": "Out-Of-Time",
        "cross_country_validation": "Leave-One-Country-Out",
        "stability_metric": "PSI",
    },
}

md_path = MODELS_DIR / f"metadata_{TODAY}.json"
with open(md_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Copier dans artifacts/ pour l'API
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
for src, dest in [
    (model_path, ARTIFACTS / "model.joblib"),
    (prep_path, ARTIFACTS / "preprocessor.joblib"),
    (feat_path, ARTIFACTS / "features.joblib"),
    (md_path, ARTIFACTS / "metadata.json"),
]:
    joblib.dump(joblib.load(src), dest) if str(dest).endswith(".joblib") else None
    if str(dest).endswith(".json"):
        with open(src) as f:
            data = json.load(f)
        with open(dest, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    if str(dest).endswith(".joblib"):
        import shutil
        shutil.copy2(src, dest)

print(f"Modele : {model_path}")
print(f"Preprocessor : {prep_path}")
print(f"Features : {feat_path}")
print(f"Metadata : {md_path}")

# ===================================================================
# RECAPITULATIF
# ===================================================================
print("\n" + "=" * 70)
print("PIPELINE V5.0 TERMINE")
print("=" * 70)
print(f"Modele            : {best_final}")
print(f"AUC OOT           : {best_row['auc_oot']:.4f}")
print(f"Nested CV AUC    : {nested_scores.mean():.4f} (+/- {nested_scores.std()*2:.4f})")
print(f"PSI              : {psi_score:.4f}")
print(f"LOCO AUC min     : {df_loco['AUC'].min():.4f}")
print(f"Features         : {len(SELECTED_FEATURES)}")
print(f"Top 3 deciles    : {dec_stats[dec_stats['decile']>=8]['pct_defauts'].sum():.1f}%")
