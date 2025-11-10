"""Tests unitaires pour le module ECL."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.evaluation.ecl import calculate_lgd, calculate_ead, calculate_ecl, calculate_discount_factor

def test_lgd_avec_garantie():
    df = pd.DataFrame({
        "has_collateral": [1], "collateral_value": [5000000],
        "collateral_type": ["Real_estate"], "loan_amount": [10000000],
        "term_months": [36]
    })
    lgd = calculate_lgd(df)
    assert 0.10 <= lgd.iloc[0] <= 0.90, "LGD doit etre entre 0.10 et 0.90"

def test_lgd_sans_garantie():
    df = pd.DataFrame({
        "has_collateral": [0], "collateral_value": [0],
        "collateral_type": [None], "loan_amount": [5000000],
        "term_months": [24]
    })
    lgd = calculate_lgd(df)
    assert abs(lgd.iloc[0] - 0.45) < 0.01, "LGD sans garantie = 45% (BCEAO)"

def test_ead_positive():
    df = pd.DataFrame({"loan_amount": [10000000], "term_months": [36]})
    ead = calculate_ead(df)
    assert ead.iloc[0] > 0, "EAD doit etre positif"
    assert ead.iloc[0] <= 10000000, "EAD ne doit pas depasser le montant du pret"

def test_ecl_calculation():
    df = pd.DataFrame({
        "has_collateral": [1, 0], "collateral_value": [5000000, 0],
        "collateral_type": ["Real_estate", None], "loan_amount": [10000000, 5000000],
        "term_months": [36, 24]
    })
    pd_vals = np.array([0.05, 0.25])
    result, summary = calculate_ecl(df, pd_vals)
    assert "ECL" in result.columns
    assert "PD" in result.columns
    assert result["ECL"].iloc[1] > result["ECL"].iloc[0], "PD plus elevee = ECL plus elevee"

def test_discount_factor():
    df = pd.Series([12, 24, 36])
    df = calculate_discount_factor(df, rate=0.05)
    assert df.iloc[0] > df.iloc[-1], "Maturite courte = DF plus eleve"
