"""
Generateur de donnees synthetiques — Credit Scoring UEMOA.
Genere credit_scoring_ucb.csv ET macro_uemoa.csv.
Le defaut depend des variables macro → les scenarios IFRS 9 sont coherents.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os

np.random.seed(42)
N = 50000
ROOT = Path(__file__).resolve().parents[2]
OUT_RAW = ROOT / "data" / "raw"; OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_EXT = ROOT / "data" / "external"; OUT_EXT.mkdir(parents=True, exist_ok=True)

# ===================== MACRO =====================
macro_dates = pd.date_range("2020-01-01", "2024-12-31", freq="ME")
n_quarters = len(macro_dates)
# Tendance legerement croissante + bruit
trend = np.linspace(0, 0.8, n_quarters)
macro = pd.DataFrame({
    "date": macro_dates,
    "gdp_growth": 3.0 - trend + np.random.normal(0, 1.2, n_quarters),
    "inflation": 2.5 + trend * 0.5 + np.random.normal(0, 0.7, n_quarters),
    "unemployment": 6.0 + trend * 0.3 + np.random.normal(0, 0.5, n_quarters),
    "interbank_rate": 4.0 + trend * 0.4 + np.random.normal(0, 0.6, n_quarters),
})

# ===================== DEMOGRAPHIE =====================
countries = ["CIV", "SEN", "MLI", "BFA", "TGO", "BEN"]
country_probs = [0.30, 0.18, 0.17, 0.15, 0.10, 0.10]
df = pd.DataFrame({
    "country": np.random.choice(countries, N, p=country_probs),
    "application_date": pd.date_range("2020-01-01", "2024-12-31", periods=N)
    + pd.to_timedelta(np.random.randint(0, 30, N), unit="D"),
})
df["application_date"] = pd.to_datetime(df["application_date"])

age_loc = {"CIV": 34, "SEN": 33, "MLI": 32, "BFA": 31, "TGO": 33, "BEN": 32}
df["age"] = np.array([max(21, int(np.random.normal(age_loc[c], 10))) for c in df["country"]]).clip(21, 75)
df["gender"] = np.random.choice(["M", "F"], N, p=[0.58, 0.42])

def marital(age):
    p_single = 0.5 if age < 30 else 0.2 if age < 45 else 0.05
    return np.random.choice(["Single", "Married", "Divorced/Widowed"], p=[p_single, 1 - p_single - 0.1, 0.1])
df["marital_status"] = [marital(a) for a in df["age"]]

df["education"] = np.random.choice(["Primary", "Secondary", "Bachelor", "Master"], N, p=[0.25, 0.35, 0.25, 0.15])
df["employment_type"] = np.random.choice(["Formal", "Informal", "Self-employed", "Civil_servant"], N, p=[0.25, 0.40, 0.20, 0.15])
df["sector"] = np.random.choice(["Agriculture", "Commerce", "Construction", "Transport", "Services", "Education", "Finance", "Telecom"], N)

# ===================== MERGE MACRO (par trimestre, sans doublons) =====================
df["_quarter"] = df["application_date"].dt.to_period("Q")
macro["_quarter"] = macro["date"].dt.to_period("Q")
# Garder une seule ligne par trimestre (derniere valeur du trimestre)
macro_quarterly = macro.groupby("_quarter").last().reset_index()
df = df.merge(macro_quarterly.drop(columns=["date"]), on="_quarter", how="left")
df = df.drop(columns=["_quarter"])

# ===================== REVENUS =====================
base_income = {"Formal": 250000, "Informal": 80000, "Self-employed": 120000, "Civil_servant": 350000}
country_mult = {"CIV": 1.2, "SEN": 1.1, "MLI": 0.8, "BFA": 0.7, "TGO": 0.8, "BEN": 0.85}
df["income_monthly"] = np.array([
    max(30000, np.random.lognormal(np.log(base_income[e] * country_mult[c] * (1 + 0.01 * (a - 30))), 0.5))
    for e, c, a in zip(df["employment_type"], df["country"], df["age"])
])
df["income_annual"] = df["income_monthly"] * 12
df["dependents"] = np.random.poisson(2, N).clip(0, 8)
df["expenses_monthly"] = (df["income_monthly"] * np.random.beta(4, 2, N) * 0.7).astype(int).clip(lower=20000)

# ===================== ENDETTEMENT =====================
df["existing_debt"] = np.where(np.random.random(N) < 0.5, 0, (df["income_monthly"] * np.random.lognormal(0.5, 1.2, N)).astype(int))
df["monthly_debt_payment"] = (df["existing_debt"] * 0.03).clip(0, df["income_monthly"] * 0.4).astype(int)

# ===================== PRET =====================
df["loan_amount"] = (df["income_monthly"] * np.random.uniform(2, 15, N)).astype(int)
df["loan_purpose"] = np.random.choice(["Working_capital", "Equipment", "Vehicle", "Real_estate", "Consumption", "Agriculture"], N, p=[0.25, 0.15, 0.15, 0.10, 0.25, 0.10])
df["term_months"] = np.random.choice([6, 12, 18, 24, 36, 48, 60], N, p=[0.05, 0.15, 0.10, 0.25, 0.25, 0.15, 0.05])
df["interest_rate"] = 8 + np.random.gamma(2, 2, N)
r_m = df["interest_rate"] / 100 / 12; t = df["term_months"]
df["monthly_payment"] = (df["loan_amount"] * r_m * (1 + r_m)**t / ((1 + r_m)**t - 1)).fillna(df["loan_amount"] / df["term_months"] * (1 + df["interest_rate"] / 100))

# ===================== GARANTIE =====================
df["has_collateral"] = np.where(np.random.random(N) < 0.55, 1, 0)
df["collateral_type"] = np.where(df["has_collateral"] == 1, np.random.choice(["Inventory", "Real_estate", "Vehicle", "Equipment", "Cash_deposit"], N), None)
df["collateral_value"] = np.where(df["has_collateral"] == 1, (df["loan_amount"] * np.random.uniform(0.8, 3.0, N)).astype(int), 0)
df["loan_to_value"] = np.where(df["collateral_value"] > 0, df["loan_amount"] / df["collateral_value"], np.nan)

# ===================== ACTIFS =====================
df["total_assets"] = (df["income_monthly"] * np.random.lognormal(2, 1.5, N)).astype(int)
df["savings_balance"] = np.where(np.random.random(N) < 0.3, 0, (df["income_monthly"] * np.random.lognormal(1, 2, N)).astype(int))
df["checking_avg_balance"] = (df["income_monthly"] * np.random.uniform(0.1, 2, N)).astype(int)

# ===================== RATIOS =====================
df["debt_to_income"] = ((df["monthly_debt_payment"] + df["monthly_payment"]) / (df["income_monthly"] + 1)).clip(0, 3)
df["expense_to_income"] = (df["expenses_monthly"] / (df["income_monthly"] + 1)).clip(0, 1.5)
df["savings_to_income"] = (df["savings_balance"] / (df["income_monthly"] + 1)).clip(0, 20)
df["payment_to_income"] = (df["monthly_payment"] / (df["income_monthly"] + 1)).clip(0, 2)

# ===================== BANCAIRE =====================
df["tenure_months"] = np.random.lognormal(3.5, 0.8, N).astype(int).clip(1, 240)
df["n_products"] = np.random.poisson(2, N).clip(0, 8) + 1
df["has_savings_account"] = (df["savings_balance"] > 0).astype(int)
df["has_insurance"] = np.random.binomial(1, 0.3, N)
df["transaction_freq_monthly"] = np.random.poisson(8, N).clip(1, 60)
df["avg_transaction_amount"] = (df["income_monthly"] * np.random.uniform(0.01, 0.3, N)).astype(int)
df["total_transactions_12m"] = (df["transaction_freq_monthly"] * 12 + np.random.poisson(5, N)).clip(1, 500)
df["bounced_checks_12m"] = np.random.poisson(0.3, N).clip(0, 5)
df["overdrafts_12m"] = np.random.poisson(1, N).clip(0, 10)
df["uses_mobile_banking"] = np.random.binomial(1, 0.6, N)
df["email_verified"] = np.random.binomial(1, 0.7, N)

# ===================== BUREAU DE CREDIT =====================
df["bureau_score"] = (
    580 + 80 * (np.log(df["income_monthly"] + 1) - np.log(df["income_monthly"] + 1).mean()) / np.log(df["income_monthly"] + 1).std()
    + 20 * (df["age"] - df["age"].mean()) / df["age"].std()
    - 60 * (df["debt_to_income"] - df["debt_to_income"].mean()) / df["debt_to_income"].std()
    + np.random.normal(0, 70, N)
).clip(350, 850).astype(int)

df["inquiries_6m"] = np.random.poisson(np.clip(6 - df["bureau_score"] / 100, 0, 8), N).clip(0, 12)
df["inquiries_3m"] = (df["inquiries_6m"] * np.random.uniform(0.3, 0.7, N)).astype(int)
df["total_credit_lines"] = np.random.poisson(np.clip(df["age"] / 8, 1, 15), N).clip(0, 20)
df["open_credit_lines"] = (df["total_credit_lines"] * np.random.uniform(0.3, 0.9, N)).astype(int)
df["credit_utilization"] = (np.random.beta(2, 3 + df["bureau_score"] / 200, N) * 100).clip(0, 100)
df["derogatory_marks"] = np.random.poisson(np.clip(8 - df["bureau_score"] / 150, 0, 6), N).clip(0, 10)
df["oldest_credit_months"] = (df["age"] * 12 * np.random.uniform(0.3, 0.9, N)).astype(int).clip(12, 700)
df["avg_credit_age_months"] = (df["oldest_credit_months"] * np.random.uniform(0.4, 0.9, N)).astype(int)
df["bureau_data_available"] = True

# ===================== INCIDENTS PAIEMENT =====================
dpd_base = np.clip(180 - df["bureau_score"] / 3 + df["debt_to_income"] * 60 + np.random.normal(0, 40, N), 0, 200).astype(int)
df["dpd_max_24m"] = dpd_base
df["n_dpd_30_12m"] = (dpd_base / 30 * np.random.uniform(0.3, 0.8, N)).astype(int).clip(0, 12)
df["years_employed"] = np.where(df["employment_type"] == "Self-employed", np.random.lognormal(1.5, 0.7, N), np.random.lognormal(1.8, 0.5, N)).clip(0.5, df["age"] - 18)

# ===================== CIBLE : DEFAUT (avec effet MACRO) =====================
def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-9)

logit = (
    -2.9
    - 0.90 * zscore(df["bureau_score"])
    + 0.60 * zscore(df["debt_to_income"].clip(0, 2))
    + 0.55 * zscore(np.log1p(df["dpd_max_24m"]))
    + 0.35 * zscore(df["payment_to_income"].clip(0, 1.5))
    + 0.25 * zscore(df["inquiries_6m"])
    + 0.30 * zscore(df["credit_utilization"])
    - 0.25 * zscore(df["age"])
    - 0.35 * zscore(np.log1p(df["income_monthly"]))
    + 0.18 * zscore(np.log1p(df["derogatory_marks"]))
    - 0.20 * zscore(np.log1p(df["tenure_months"]))
    + 0.12 * zscore(np.log1p(df["bounced_checks_12m"]))
)

# === EFFETS MACRO-ECONOMIQUES (ce qui rend les scenarios coherents) ===
# Coefficients recalibres pour scenarios economiquement plausibles (ref: BCEAO 2008-2020)
# Chaque variable macro influence directement le logit de defaut
logit += (
    - 0.15 * zscore(df["gdp_growth"])          # Croissance ↓ → defaut ↑
    + 0.12 * zscore(df["inflation"])            # Inflation ↑ → defaut ↑
    + 0.10 * zscore(df["unemployment"])         # Chomage ↑ → defaut ↑
    + 0.08 * zscore(df["interbank_rate"])       # Taux ↑ → defaut ↑
)

# Interactions macro × profil client
logit += 0.15 * zscore(df["debt_to_income"].clip(0, 2)) * zscore(df["inflation"])   # DTI × inflation
logit += 0.12 * zscore(df["dpd_max_24m"] / 30 + 1) * zscore(df["unemployment"])     # DPD × chomage
logit += 0.10 * zscore(df["term_months"]) * zscore(df["interbank_rate"])             # Terme × taux

# Interactions bureau/DPD (existantes)
logit += 0.50 * (df["bureau_score"] < 550).astype(float) * (df["debt_to_income"] > 0.8).astype(float)
logit += 0.35 * (df["age"] < 28).astype(float) * (df["total_credit_lines"] < 3).astype(float)

# Bruit
logit += np.random.normal(0, 0.65, N)

df["default_probability"] = 1 / (1 + np.exp(-logit))
df["default"] = (np.random.random(N) < df["default_probability"]).astype(int)

# ===================== VERIFICATION =====================
from sklearn.metrics import roc_auc_score
df_clean = df.dropna(subset=["default_probability"])
print(f"Taux defaut: {df_clean['default'].mean()*100:.1f}%")
print(f"AUC theorique: {roc_auc_score(df_clean['default'], df_clean['default_probability']):.4f}")
print(f"Corr bureau x default: {df_clean['bureau_score'].corr(df_clean['default']):.4f}")
print(f"Corr GDP x default: {df_clean['gdp_growth'].corr(df_clean['default']):.4f}")
print(f"Corr inflation x default: {df_clean['inflation'].corr(df_clean['default']):.4f}")
print(f"Corr unemployment x default: {df_clean['unemployment'].corr(df_clean['default']):.4f}")

# ===================== VERIFICATION SCENARIOS =====================
# Simuler 3 scenarios pour verifier la monotonie
df_m = df_clean.copy()
def check_scenarios():
    """Verification rapide de la monotonie des scenarios IFRS 9.

    Les chocs en z-score correspondent aux chocs bruts definis dans macro.py :
      adverse:  gdp -1.5pp, infl +2.0pp, unemp +1.5pp, taux +1.5pp
      severe:   gdp -3.5pp, infl +4.0pp, unemp +3.0pp, taux +3.0pp
    """
    baseline = df_m["default_probability"].mean()
    g_z = zscore(df_m["gdp_growth"]); i_z = zscore(df_m["inflation"])
    u_z = zscore(df_m["unemployment"]); r_z = zscore(df_m["interbank_rate"])
    # Adverse: chocs convertis en unites de z-score (ecart-type)
    gdp_z_adv = g_z.mean() - 1.25 / df_m["gdp_growth"].std()
    infl_z_adv = i_z.mean() + 3.0 / df_m["inflation"].std()
    unemp_z_adv = u_z.mean() + 3.0 / df_m["unemployment"].std()
    rate_z_adv = r_z.mean() + 2.5 / df_m["interbank_rate"].std()

    adverse_logit = logit.copy()[:len(df_m)]
    adverse_logit += (-0.15 * (gdp_z_adv - g_z)
                      + 0.12 * (infl_z_adv - i_z)
                      + 0.10 * (unemp_z_adv - u_z)
                      + 0.08 * (rate_z_adv - r_z))
    adverse_pd = 1 / (1 + np.exp(-adverse_logit))

    # Severe: chocs convertis en unites de z-score
    gdp_z_sev = g_z.mean() - 2.9 / df_m["gdp_growth"].std()
    infl_z_sev = i_z.mean() + 5.7 / df_m["inflation"].std()
    unemp_z_sev = u_z.mean() + 6.0 / df_m["unemployment"].std()
    rate_z_sev = r_z.mean() + 5.0 / df_m["interbank_rate"].std()

    severe_logit = logit.copy()[:len(df_m)]
    severe_logit += (-0.15 * (gdp_z_sev - g_z)
                     + 0.12 * (infl_z_sev - i_z)
                     + 0.10 * (unemp_z_sev - u_z)
                     + 0.08 * (rate_z_sev - r_z))
    severe_pd = 1 / (1 + np.exp(-severe_logit))

    print(f"\nVerification scenarios IFRS 9:")
    print(f"  PD baseline: {baseline*100:.2f}%")
    print(f"  PD adverse:  {adverse_pd.mean()*100:.2f}%")
    print(f"  PD severe:   {severe_pd.mean()*100:.2f}%")
    ok = (baseline < adverse_pd.mean()) and (adverse_pd.mean() < severe_pd.mean())
    print(f"  Monotonie PD_baseline < PD_adverse < PD_severe: {'OK' if ok else 'KO'}")
    if severe_pd.mean() > 0.60:
        print(f"  ATTENTION: PD severe > 60% ({severe_pd.mean()*100:.1f}%) — scenarios irrealistes")

check_scenarios()

# ===================== VALEURS MANQUANTES =====================
for col in ["education", "marital_status"]:
    df.loc[np.random.random(N) < 0.05, col] = None
mask_informal = df["employment_type"] == "Informal"
df.loc[mask_informal & (np.random.random(N) < 0.25), ["income_monthly", "income_annual", "expenses_monthly"]] = np.nan
mask_high_debt = df["existing_debt"] > df["income_monthly"] * 5
df.loc[mask_high_debt & (np.random.random(N) < 0.20), ["existing_debt", "monthly_debt_payment"]] = np.nan
mask_no_bureau = np.random.random(N) < 0.15
for col in ["bureau_score", "inquiries_6m", "inquiries_3m", "total_credit_lines", "open_credit_lines", "credit_utilization", "derogatory_marks", "oldest_credit_months", "avg_credit_age_months"]:
    df.loc[mask_no_bureau, col] = np.nan
df.loc[mask_no_bureau, "bureau_data_available"] = False
df.loc[np.random.random(N) < 0.30, "checking_avg_balance"] = np.nan
df.loc[df["has_collateral"] == 0, "collateral_type"] = None
df.loc[df["has_collateral"] == 0, "loan_to_value"] = np.nan

# ===================== SAUVEGARDE =====================
df["customer_id"] = [f"UCB{i+1:07d}" for i in range(N)]

cols_final = [
    "customer_id", "application_date", "country", "age", "gender",
    "marital_status", "education", "employment_type", "years_employed", "sector",
    "income_monthly", "income_annual", "expenses_monthly",
    "existing_debt", "monthly_debt_payment", "dependents",
    "total_assets", "savings_balance", "checking_avg_balance",
    "debt_to_income", "expense_to_income", "savings_to_income",
    "loan_amount", "loan_purpose", "term_months", "interest_rate",
    "monthly_payment", "has_collateral", "collateral_type", "collateral_value",
    "loan_to_value", "payment_to_income",
    "tenure_months", "n_products", "has_savings_account", "has_insurance",
    "transaction_freq_monthly", "avg_transaction_amount", "total_transactions_12m",
    "bounced_checks_12m", "overdrafts_12m",
    "dpd_max_24m", "n_dpd_30_12m",
    "uses_mobile_banking", "email_verified",
    "bureau_score", "inquiries_6m", "inquiries_3m",
    "total_credit_lines", "open_credit_lines", "credit_utilization",
    "derogatory_marks", "oldest_credit_months", "avg_credit_age_months",
    "bureau_data_available",
    # Variables macro mergées (pour forward-looking IFRS 9)
    "gdp_growth", "inflation", "unemployment", "interbank_rate",
    "default", "default_probability",
]
# Garder seulement les colonnes existantes
cols_final = [c for c in cols_final if c in df.columns]

df[cols_final].to_csv(OUT_RAW / "credit_scoring_ucb.csv", index=False)
macro.drop(columns=["_quarter"]).to_csv(OUT_EXT / "macro_uemoa.csv", index=False)

print(f"\nDonnees generees:")
print(f"  data/raw/credit_scoring_ucb.csv : {df.shape[0]:,} x {len(cols_final)}")
print(f"  data/external/macro_uemoa.csv  : {len(macro):,} trimestres")
