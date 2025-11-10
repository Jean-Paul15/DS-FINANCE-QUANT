"""
Inference Production - Scoring Credit UCB
==========================================
Compatible avec EBM, LogReg, XGBoost, LightGBM, RandomForest.

Utilisation:
    scorer = CreditScorer(
        model_path='outputs/models/credit_scoring_EBM_*.joblib',
        preprocessor_path='outputs/models/preprocessor_*.joblib',
        features_path='outputs/models/features_*.joblib',
        metadata_path='outputs/models/metadata_*.json'
    )
    results = scorer.predict_full(df_nouveaux_clients)
"""
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Fonction necessaire au deserialisation du preprocessor
# (utilisee par FunctionTransformer dans le pipeline categoriel)
def _df_to_object(X):
    return X.astype(object)
import sys as _sys
_sys.modules['__main__']._df_to_object = _df_to_object


def feature_engineering(df, train_stats=None):
    """Feature engineering cible - 26 features basees sur l'EDA.

    Parameters
    ----------
    df : DataFrame
    train_stats : dict or None
        Medianes du train, necessaires pour l'inference mono-client.
    """
    df = df.copy()

    # ---- Indicateurs bureau ----
    df['bureau_low'] = (df['bureau_score'].fillna(620) < 550).astype(int)
    df['bureau_medium'] = ((df['bureau_score'].fillna(620) >= 550) &
                           (df['bureau_score'].fillna(620) < 700)).astype(int)
    df['bureau_high'] = (df['bureau_score'].fillna(620) >= 700).astype(int)
    df['bureau_missing'] = df['bureau_score'].isna().astype(int)

    # ---- Severite paiement ----
    df['dpd_severity'] = (df['dpd_max_24m'].fillna(0) / 30 +
                          df['n_dpd_30_12m'].fillna(0) * 0.5 +
                          df['bounced_checks_12m'].fillna(0) * 0.3)
    df['has_payment_issues'] = ((df['dpd_max_24m'].fillna(0) > 0) |
                                (df['bounced_checks_12m'].fillna(0) > 0)).astype(int)

    # ---- Capacite remboursement ----
    if train_stats:
        inc_med = train_stats["inc_med"]
        exp_med = train_stats["exp_med"]
    else:
        inc_med = df['income_monthly'].median() if df['income_monthly'].notna().any() else 250000
        exp_med = df['expenses_monthly'].median() if df['expenses_monthly'].notna().any() else 100000
    df['net_income'] = df['income_monthly'].fillna(inc_med) - df['expenses_monthly'].fillna(exp_med)
    df['repayment_capacity'] = (df['net_income'] -
                                df['monthly_payment'].fillna(0) -
                                df['monthly_debt_payment'].fillna(0))
    df['repayment_ratio'] = df['repayment_capacity'] / (df['income_monthly'].fillna(inc_med) + 1)
    df['total_burden'] = ((df['monthly_payment'].fillna(0) +
                           df['monthly_debt_payment'].fillna(0)) /
                          (df['income_monthly'].fillna(inc_med) + 1))

    # ---- Stabilite ----
    df['stable_employment'] = (df['years_employed'].fillna(0) >= 3).astype(int)
    df['long_tenure'] = (df['tenure_months'].fillna(0) >= 36).astype(int)
    df['engagement_score'] = (df['n_products'].fillna(1) +
                              df['has_savings_account'].fillna(0) +
                              df['has_insurance'].fillna(0))

    # ---- Ratios et interactions ----
    if train_stats:
        exp_med_ratio = train_stats["exp_med"]
        dti_med = train_stats["dti_med"]
    else:
        exp_med_ratio = df['expenses_monthly'].median() if df['expenses_monthly'].notna().any() else 100000
        dti_med = df['debt_to_income'].median() if df['debt_to_income'].notna().any() else 0.3
    df['asset_coverage'] = df['total_assets'].fillna(0) / (df['loan_amount'] + 1)
    df['liquidity_ratio'] = ((df['savings_balance'].fillna(0) +
                              df['checking_avg_balance'].fillna(0)) /
                             (df['expenses_monthly'].fillna(exp_med_ratio) + 1))
    bureau_fill = df['bureau_score'].fillna(620)
    dti_fill = df['debt_to_income'].fillna(dti_med)
    df['bureau_x_dti'] = bureau_fill * dti_fill
    df['dpd_x_dti'] = df['dpd_severity'] * dti_fill

    # ---- Log transforms ----
    for col in ['income_monthly', 'loan_amount', 'existing_debt', 'total_assets']:
        if train_stats and f'{col}_med' in train_stats:
            median_val = train_stats[f'{col}_med']
        else:
            median_val = df[col].median() if df[col].notna().any() else 0
        df[f'log_{col}'] = np.log1p(df[col].fillna(median_val).clip(lower=0))

    # ---- Bins demographiques ----
    df['age_young'] = (df['age'] < 30).astype(int)
    df['age_senior'] = (df['age'] > 55).astype(int)
    df['credit_age_ratio'] = (df['avg_credit_age_months'].fillna(0) /
                              (df['age'].fillna(30) * 12 + 1))

    # ---- Score composite ----
    df['credit_risk_score'] = (
        (1 - bureau_fill / 850) * 40 +
        df['credit_utilization'].fillna(50) / 100 * 20 +
        df['inquiries_6m'].fillna(0) * 3 +
        df['derogatory_marks'].fillna(0) * 10
    )

    # ---- Pression de paiement ----
    df['payment_pressure'] = df['monthly_payment'].fillna(0) / df['net_income'].clip(lower=1)

    return df


def proba_to_score(p, base_score=500, pdo=20, base_pd=0.189):
    """Conversion probabilite de defaut -> score d'octroi (0-1000, plus haut = moins risque)."""
    base_odds = base_pd / (1 - base_pd)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds = np.clip(p / (1 - p + 1e-9), 1e-9, 1e9)
    return np.clip(1000 - (offset + factor * np.log(odds)), 0, 1000)


def classe_ifrs9(p):
    """Classification IFRS 9 par probabilite de defaut."""
    if p < 0.10:
        return 'Stage 1 - Performing'
    elif p < 0.30:
        return 'Stage 1+ - Surveillance'
    elif p < 0.50:
        return 'Stage 2 - Under-performing'
    else:
        return 'Stage 3 - Non-performing'


class CreditScorer:
    """Wrapper de production pour le scoring credit UCB.
    Compatible avec EBM, LogReg, XGBoost, LightGBM, RandomForest.
    """

    def __init__(self, model_path, preprocessor_path, features_path, metadata_path=None):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        features = joblib.load(features_path)

        # Supporte les deux formats de features (ancien et nouveau)
        if 'selected' in features:
            self.features_selected = features['selected']
        elif 'all' in features:
            self.features_selected = features['all']
        else:
            self.features_selected = features.get('selected_num', []) + features.get('selected_cat', [])

        # Charger les stats d'entrainement pour l'inference (evite median() == valeur unique)
        self.train_stats = features.get('train_stats', None)

        self.metadata = None
        if metadata_path:
            with open(metadata_path, encoding='utf-8') as f:
                self.metadata = json.load(f)

        self.is_ebm = 'ExplainableBoosting' in type(self.model).__name__

    def predict_proba(self, df_raw):
        """Retourne les probabilites de defaut.

        Args:
            df_raw: DataFrame avec donnees clients brutes (avant feature engineering)
        Returns:
            array probabilites de defaut
        """
        df_fe = feature_engineering(df_raw, train_stats=self.train_stats)
        X = df_fe[self.features_selected]
        X_proc = self.preprocessor.transform(X)

        if self.is_ebm:
            X_proc_df = pd.DataFrame(X_proc, columns=self.features_selected)
            return self.model.predict_proba(X_proc_df)[:, 1]
        return self.model.predict_proba(X_proc)[:, 1]

    def predict_full(self, df_raw):
        """Retourne PD, score d'octroi et classification IFRS 9.

        Args:
            df_raw: DataFrame avec donnees clients brutes
        Returns:
            DataFrame avec customer_id, PD_estimee, score_octroi, classe_ifrs9
        """
        probas = self.predict_proba(df_raw)
        scores = proba_to_score(probas).astype(int)
        classes = [classe_ifrs9(p) for p in probas]

        result = pd.DataFrame({
            'customer_id': df_raw['customer_id'].values if 'customer_id' in df_raw.columns
                           else [f'CLIENT{i+1:06d}' for i in range(len(df_raw))],
            'PD_estimee': probas.round(4),
            'score_octroi': scores,
            'classe_ifrs9': classes
        })
        return result

    def explain(self, df_raw, top_n=10):
        """Explication individuelle EBM (pour audit reglementaire).

        Args:
            df_raw: DataFrame avec donnees client (1 ligne recommandee)
            top_n: Nombre de features a afficher
        Returns:
            DataFrame avec features et contributions
        """
        if not self.is_ebm:
            print("Explication disponible uniquement pour EBM. Utilisez SHAP/LIME pour les autres modeles.")
            return None

        df_fe = feature_engineering(df_raw.head(1), train_stats=self.train_stats)
        X = df_fe[self.features_selected]
        X_proc = self.preprocessor.transform(X)
        X_proc_df = pd.DataFrame(X_proc, columns=self.features_selected)

        local_exp = self.model.explain_local(X_proc_df)
        data = local_exp.data(0)

        contributions = pd.DataFrame({
            'feature': data['names'],
            'contribution': data['scores']
        })
        contributions = contributions.reindex(
            contributions['contribution'].abs().sort_values(ascending=False).index
        ).head(top_n)
        return contributions

    def predict_ecl(self, df_raw):
        """Calcule l'ECL complete (PD × LGD × EAD × DF) selon IFRS 9.

        Args:
            df_raw: DataFrame avec donnees clients brutes
        Returns:
            DataFrame avec PD, LGD, EAD, DF, ECL, stage IFRS 9
        """
        try:
            from src.evaluation.ecl import calculate_lgd, calculate_ead, calculate_ecl
        except ImportError:
            print("Module ECL non disponible. Verifier src/evaluation/ecl.py")
            return None

        probas = self.predict_proba(df_raw)
        lgd = calculate_lgd(df_raw, method="collateral_based")
        ead = calculate_ead(df_raw)
        result, summary = calculate_ecl(df_raw, probas, lgd, ead)

        # Ajouter score et classification
        scores = proba_to_score(probas).astype(int)
        result["score"] = scores
        result["classe_ifrs9"] = [classe_ifrs9(p) for p in probas]
        if "customer_id" in df_raw.columns:
            result["customer_id"] = df_raw["customer_id"].values

        return result


if __name__ == '__main__':
    import sys

    # Utiliser outputs/models/ comme chemin par defaut
    models_dir = Path('outputs/models')
    if not models_dir.exists():
        models_dir = Path('models')

    if not models_dir.exists():
        print(f"Repertoire modeles introuvable (cherche: models/ et outputs/models/)")
        sys.exit(1)

    # Auto-detect derniers fichiers
    model_files = sorted(models_dir.glob('credit_scoring_*.joblib'))
    if not model_files:
        print("Aucun modele trouve dans", models_dir)
        sys.exit(1)

    latest_model = model_files[-1]
    parts = latest_model.stem.split('_')
    # Format: credit_scoring_<MODELE>_YYYYMMDD_HHMMSS.joblib
    timestamp = '_'.join(parts[-2:])

    pp_path = models_dir / f'preprocessor_{timestamp}.joblib'
    ft_path = models_dir / f'features_{timestamp}.joblib'
    md_path = models_dir / f'metadata_{timestamp}.json'

    # Fallback si timestamp n'existe pas (fichiers renommes)
    if not pp_path.exists():
        pp_files = sorted(models_dir.glob('preprocessor_*.joblib'))
        if pp_files:
            pp_path = pp_files[-1]
    if not ft_path.exists():
        ft_files = sorted(models_dir.glob('features_*.joblib'))
        if ft_files:
            ft_path = ft_files[-1]
    if not md_path.exists():
        md_files = sorted(models_dir.glob('metadata_*.json'))
        if md_files:
            md_path = md_files[-1]

    scorer = CreditScorer(
        model_path=str(latest_model),
        preprocessor_path=str(pp_path),
        features_path=str(ft_path),
        metadata_path=str(md_path) if md_path.exists() else None
    )

    if scorer.metadata:
        print(f"Modele : {scorer.metadata['model_info']['name']}")
        print(f"AUC OOT: {scorer.metadata['performance']['auc_oot']:.4f}")
        print(f"Features: {scorer.metadata['features']['n_selected']}")
    else:
        print(f"Modele charge : {type(scorer.model).__name__}")

    # Test sur les premiers clients
    data_path = Path('data/raw/credit_scoring_ucb.csv')
    if data_path.exists():
        df_test = pd.read_csv(data_path).head(10)
        results = scorer.predict_full(df_test)
        print("\nResultats inference (10 premiers clients) :")
        print(results.to_string(index=False))
    else:
        print("\nFichier data/raw/credit_scoring_ucb.csv introuvable pour test.")
