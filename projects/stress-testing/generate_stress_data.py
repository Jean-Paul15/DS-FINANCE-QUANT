#!/usr/bin/env python3
"""
=============================================================================
STRESS TESTING DATA GENERATOR - SYNTHETIC LOAN PORTFOLIO
=============================================================================
Génère des données pour le stress testing macroéconomique :
- Données macroéconomiques UEMOA (2000-2024)
- Portefeuille de prêts avec historique de défauts
- Relation calibrée entre variables macro et taux de défaut

Usage: python generate_stress_data.py
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'start_year': 2000,
    'end_year': 2024,
    'n_loans': 50000,
    'countries': ['SEN', 'CIV', 'MLI', 'BFA', 'TGO', 'BEN', 'CMR', 'GAB'],
}

# =============================================================================
# MACRO DATA GENERATION
# =============================================================================

def generate_macro_data():
    """Generate realistic macroeconomic data for UEMOA zone"""
    print("Generating macroeconomic data...")
    
    years = range(CONFIG['start_year'], CONFIG['end_year'] + 1)
    n_periods = len(years) * 4  # Quarterly data
    
    dates = pd.date_range(
        start=f"{CONFIG['start_year']}-01-01",
        end=f"{CONFIG['end_year']}-12-31",
        freq='Q'
    )[:n_periods]
    
    # Base trends with cycles
    t = np.arange(n_periods)
    
    # GDP Growth (%)
    gdp_trend = 4.5  # Long-term trend
    gdp_cycle = 2 * np.sin(2 * np.pi * t / 40)  # ~10 year cycle
    gdp_noise = np.random.normal(0, 0.8, n_periods)
    
    # Crisis impacts
    gdp_crisis = np.zeros(n_periods)
    # 2008-2009 crisis
    crisis_2008_start = (2008 - CONFIG['start_year']) * 4
    gdp_crisis[crisis_2008_start:crisis_2008_start+6] = [-2, -4, -5, -4, -2, 0][:min(6, n_periods-crisis_2008_start)]
    # 2020 COVID
    covid_start = (2020 - CONFIG['start_year']) * 4
    if covid_start < n_periods:
        gdp_crisis[covid_start:covid_start+4] = [-3, -6, -4, -1][:min(4, n_periods-covid_start)]
    
    gdp_growth = gdp_trend + gdp_cycle + gdp_crisis + gdp_noise
    gdp_growth = np.clip(gdp_growth, -8, 12)
    
    # Inflation (%)
    inflation_base = 2.5
    inflation_cycle = 1.5 * np.sin(2 * np.pi * t / 20 + np.pi/4)
    inflation_noise = np.random.normal(0, 0.5, n_periods)
    # Oil shock impact
    inflation_shock = np.zeros(n_periods)
    oil_shock_2008 = (2008 - CONFIG['start_year']) * 4 + 2
    if oil_shock_2008 < n_periods:
        inflation_shock[oil_shock_2008:oil_shock_2008+4] = [2, 3, 2, 1][:min(4, n_periods-oil_shock_2008)]
    oil_shock_2022 = (2022 - CONFIG['start_year']) * 4
    if oil_shock_2022 < n_periods:
        inflation_shock[oil_shock_2022:oil_shock_2022+6] = [1, 2, 3, 3, 2, 1][:min(6, n_periods-oil_shock_2022)]
    
    inflation = inflation_base + inflation_cycle + inflation_shock + inflation_noise
    inflation = np.clip(inflation, 0.5, 12)
    
    # Policy Rate (BCEAO) - responds to inflation with lag
    policy_rate_base = 3.5
    policy_rate = policy_rate_base + 0.4 * (inflation - 2.5)
    policy_rate = np.clip(policy_rate, 2.0, 7.0)
    
    # Unemployment (%)
    unemployment_base = 7.0
    unemployment_cycle = -0.8 * gdp_cycle  # Counter-cyclical
    unemployment_noise = np.random.normal(0, 0.3, n_periods)
    unemployment = unemployment_base + unemployment_cycle - 0.3 * gdp_growth + unemployment_noise
    unemployment = np.clip(unemployment, 4, 15)
    
    # Oil Price (USD/barrel)
    oil_base = 60
    oil_trend = 0.5 * t  # Gradual increase
    oil_cycle = 20 * np.sin(2 * np.pi * t / 32)
    oil_noise = np.random.normal(0, 5, n_periods)
    oil_shock = np.zeros(n_periods)
    if oil_shock_2008 < n_periods:
        oil_shock[oil_shock_2008-2:oil_shock_2008+2] = [20, 40, 30, 10][:min(4, n_periods-oil_shock_2008+2)]
    if oil_shock_2022 < n_periods:
        oil_shock[oil_shock_2022:oil_shock_2022+4] = [20, 40, 30, 20][:min(4, n_periods-oil_shock_2022)]
    
    oil_price = oil_base + oil_trend/4 + oil_cycle + oil_shock + oil_noise
    oil_price = np.clip(oil_price, 20, 150)
    
    # Credit Growth (%)
    credit_growth = 8 + 0.5 * gdp_growth - 0.8 * (policy_rate - 3.5) + np.random.normal(0, 1, n_periods)
    credit_growth = np.clip(credit_growth, -5, 25)
    
    # DEFAULT RATE - Key variable, function of macro conditions
    # Higher when: low GDP, high unemployment, high rates, high inflation
    default_base = 4.0
    default_rate = (
        default_base
        - 0.4 * gdp_growth           # Lower GDP → higher default
        + 0.3 * unemployment         # Higher unemployment → higher default
        + 0.2 * (policy_rate - 3.5)  # Higher rates → higher default
        + 0.15 * (inflation - 2.5)   # Higher inflation → higher default
        + np.random.normal(0, 0.5, n_periods)
    )
    
    # Crisis amplification
    default_crisis = np.zeros(n_periods)
    if crisis_2008_start < n_periods:
        default_crisis[crisis_2008_start:crisis_2008_start+8] = [0.5, 1.5, 3, 4, 3.5, 2.5, 1.5, 0.5][:min(8, n_periods-crisis_2008_start)]
    if covid_start < n_periods:
        default_crisis[covid_start:covid_start+6] = [1, 3, 4, 3, 2, 1][:min(6, n_periods-covid_start)]
    
    default_rate = default_rate + default_crisis
    default_rate = np.clip(default_rate, 1.5, 15)
    
    macro_df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'quarter': dates.quarter,
        'gdp_growth': np.round(gdp_growth, 2),
        'inflation': np.round(inflation, 2),
        'policy_rate': np.round(policy_rate, 2),
        'unemployment': np.round(unemployment, 2),
        'oil_price': np.round(oil_price, 2),
        'credit_growth': np.round(credit_growth, 2),
        'default_rate': np.round(default_rate, 2),
    })
    
    return macro_df


# =============================================================================
# LOAN PORTFOLIO GENERATION
# =============================================================================

def generate_loan_portfolio(macro_df, n_loans):
    """Generate a loan portfolio with default history"""
    print(f"Generating loan portfolio ({n_loans:,} loans)...")
    
    # Loan IDs
    loan_ids = [f'LN{str(i).zfill(8)}' for i in range(n_loans)]
    
    # Origination dates (distributed across history)
    orig_start = pd.to_datetime('2005-01-01')
    orig_end = pd.to_datetime('2024-06-01')
    orig_days = (orig_end - orig_start).days
    origination_dates = orig_start + pd.to_timedelta(
        np.random.randint(0, orig_days, n_loans), unit='D'
    )
    
    # Loan characteristics
    loan_types = np.random.choice(
        ['corporate', 'sme', 'retail_mortgage', 'retail_consumer', 'sovereign'],
        n_loans, p=[0.15, 0.30, 0.20, 0.30, 0.05]
    )
    
    # Amount (USD) - varies by type
    amounts = np.zeros(n_loans)
    for i, lt in enumerate(loan_types):
        if lt == 'sovereign':
            amounts[i] = np.random.uniform(5_000_000, 50_000_000)
        elif lt == 'corporate':
            amounts[i] = np.random.uniform(500_000, 10_000_000)
        elif lt == 'sme':
            amounts[i] = np.random.uniform(50_000, 500_000)
        elif lt == 'retail_mortgage':
            amounts[i] = np.random.uniform(30_000, 200_000)
        else:  # retail_consumer
            amounts[i] = np.random.uniform(1_000, 30_000)
    
    # Country
    countries = np.random.choice(
        CONFIG['countries'],
        n_loans,
        p=[0.25, 0.22, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05]
    )
    
    # Interest rate at origination
    base_rates = macro_df.set_index('date')['policy_rate']
    spreads = {
        'sovereign': 1.5, 'corporate': 3.0, 'sme': 5.0,
        'retail_mortgage': 4.0, 'retail_consumer': 8.0
    }
    
    interest_rates = np.zeros(n_loans)
    for i in range(n_loans):
        orig_quarter = pd.Timestamp(origination_dates[i]).to_period('Q').to_timestamp()
        if orig_quarter in base_rates.index:
            base = base_rates.loc[orig_quarter]
        else:
            base = 3.5
        interest_rates[i] = base + spreads[loan_types[i]] + np.random.normal(0, 0.5)
    
    # Term (months)
    terms = np.zeros(n_loans)
    for i, lt in enumerate(loan_types):
        if lt == 'sovereign':
            terms[i] = np.random.choice([60, 84, 120, 180])
        elif lt == 'corporate':
            terms[i] = np.random.choice([36, 48, 60, 84])
        elif lt == 'sme':
            terms[i] = np.random.choice([24, 36, 48, 60])
        elif lt == 'retail_mortgage':
            terms[i] = np.random.choice([120, 180, 240, 300])
        else:
            terms[i] = np.random.choice([12, 24, 36, 48])
    
    # Initial PD (at origination, from scoring model)
    base_pds = {
        'sovereign': 0.02, 'corporate': 0.03, 'sme': 0.05,
        'retail_mortgage': 0.025, 'retail_consumer': 0.06
    }
    initial_pds = np.array([base_pds[lt] * np.random.uniform(0.5, 2.0) for lt in loan_types])
    initial_pds = np.clip(initial_pds, 0.005, 0.30)
    
    # LGD
    lgds = {
        'sovereign': 0.45, 'corporate': 0.45, 'sme': 0.50,
        'retail_mortgage': 0.25, 'retail_consumer': 0.60
    }
    lgd_values = np.array([lgds[lt] * np.random.uniform(0.8, 1.2) for lt in loan_types])
    lgd_values = np.clip(lgd_values, 0.10, 0.80)
    
    # Determine defaults based on macro conditions at origination + time
    defaults = np.zeros(n_loans, dtype=int)
    default_dates = [None] * n_loans
    
    for i in range(n_loans):
        orig_quarter = pd.Timestamp(origination_dates[i]).to_period('Q').to_timestamp()
        
        # Get macro conditions during loan life
        loan_end = origination_dates[i] + pd.Timedelta(days=int(terms[i]*30))
        relevant_macro = macro_df[(macro_df['date'] >= orig_quarter) & 
                                   (macro_df['date'] <= min(loan_end, macro_df['date'].max()))]
        
        if len(relevant_macro) > 0:
            avg_default_rate = relevant_macro['default_rate'].mean() / 100
            # Loan-specific probability
            default_prob = initial_pds[i] * (1 + (avg_default_rate - 0.04) * 5)
            default_prob = np.clip(default_prob, 0, 0.5)
            
            if np.random.random() < default_prob:
                defaults[i] = 1
                # Default date: random within loan life
                days_to_default = np.random.randint(90, min(int(terms[i]*30), 
                    (macro_df['date'].max() - origination_dates[i]).days))
                default_dates[i] = origination_dates[i] + pd.Timedelta(days=days_to_default)
    
    loans_df = pd.DataFrame({
        'loan_id': loan_ids,
        'origination_date': origination_dates,
        'loan_type': loan_types,
        'amount_usd': np.round(amounts, 2),
        'country': countries,
        'interest_rate': np.round(interest_rates, 2),
        'term_months': terms.astype(int),
        'initial_pd': np.round(initial_pds, 4),
        'lgd': np.round(lgd_values, 2),
        'is_default': defaults,
        'default_date': default_dates,
    })
    
    # Calculate EAD (simplified: remaining principal at default or current)
    loans_df['ead'] = loans_df['amount_usd'] * np.random.uniform(0.4, 1.0, n_loans)
    
    return loans_df


# =============================================================================
# SCENARIOS GENERATION
# =============================================================================

def generate_scenarios():
    """Generate standard stress test scenarios"""
    print("Generating stress scenarios...")
    
    # Horizons: Q1-Q4 of projection year
    horizons = ['Q1', 'Q2', 'Q3', 'Q4']
    
    scenarios = {
        'baseline': {
            'description': 'Croissance tendancielle, conditions normales',
            'gdp_growth': [4.5, 4.6, 4.7, 4.5],
            'inflation': [3.2, 3.3, 3.4, 3.5],
            'policy_rate': [3.5, 3.5, 3.5, 3.5],
            'unemployment': [6.5, 6.4, 6.3, 6.2],
            'oil_price': [80, 82, 81, 80],
        },
        'adverse': {
            'description': 'Ralentissement économique marqué',
            'gdp_growth': [2.0, 1.0, 0.5, 1.0],
            'inflation': [5.0, 5.5, 6.0, 5.5],
            'policy_rate': [4.5, 5.0, 5.0, 4.5],
            'unemployment': [8.0, 9.0, 9.5, 9.0],
            'oil_price': [100, 110, 120, 115],
        },
        'severe': {
            'description': 'Récession profonde avec stagflation',
            'gdp_growth': [-1.0, -2.5, -2.0, -0.5],
            'inflation': [7.0, 8.0, 8.5, 8.0],
            'policy_rate': [5.5, 6.0, 6.5, 6.0],
            'unemployment': [10.0, 11.5, 12.0, 11.5],
            'oil_price': [130, 145, 150, 140],
        },
        'crisis_2008': {
            'description': 'Reproduction de la crise financière 2008',
            'gdp_growth': [1.0, -1.5, -3.0, -2.0],
            'inflation': [5.5, 6.5, 7.0, 5.5],
            'policy_rate': [4.5, 5.0, 5.25, 4.5],
            'unemployment': [8.5, 10.0, 11.0, 10.5],
            'oil_price': [110, 140, 100, 60],
        },
        'covid_shock': {
            'description': 'Choc pandémique type COVID-19',
            'gdp_growth': [2.0, -6.0, -3.0, 1.0],
            'inflation': [2.5, 1.5, 2.0, 2.5],
            'policy_rate': [3.0, 2.5, 2.0, 2.0],
            'unemployment': [7.5, 11.0, 10.0, 8.5],
            'oil_price': [50, 25, 40, 50],
        },
    }
    
    # Convert to DataFrame
    scenario_dfs = {}
    for name, params in scenarios.items():
        df = pd.DataFrame({
            'scenario': name,
            'horizon': horizons,
            'gdp_growth': params['gdp_growth'],
            'inflation': params['inflation'],
            'policy_rate': params['policy_rate'],
            'unemployment': params['unemployment'],
            'oil_price': params['oil_price'],
            'description': params['description'],
        })
        scenario_dfs[name] = df
    
    all_scenarios = pd.concat(scenario_dfs.values(), ignore_index=True)
    return all_scenarios


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SYNTHETIC LOAN PORTFOLIO - STRESS TESTING DATA GENERATOR")
    print("=" * 70)
    
    # Generate data
    macro_df = generate_macro_data()
    loans_df = generate_loan_portfolio(macro_df, CONFIG['n_loans'])
    scenarios_df = generate_scenarios()
    
    # Create output directory
    output_dir = 'data_stress'
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/scenarios", exist_ok=True)
    
    # Save files
    macro_df.to_csv(f"{output_dir}/raw/macro_uemoa.csv", index=False)
    loans_df.to_csv(f"{output_dir}/raw/loan_portfolio.csv", index=False)
    scenarios_df.to_csv(f"{output_dir}/scenarios/stress_scenarios.csv", index=False)
    
    # Summary statistics
    print(f"\nFiles saved to {output_dir}/")
    print(f"  - macro_uemoa.csv: {len(macro_df)} quarters ({CONFIG['start_year']}-{CONFIG['end_year']})")
    print(f"  - loan_portfolio.csv: {len(loans_df):,} loans")
    print(f"  - stress_scenarios.csv: {len(scenarios_df)} scenario-horizons")
    
    print(f"\nPortfolio statistics:")
    print(f"  - Total exposure: ${loans_df['amount_usd'].sum()/1e6:.1f}M")
    print(f"  - Default rate: {loans_df['is_default'].mean()*100:.2f}%")
    print(f"  - By type: {loans_df.groupby('loan_type')['is_default'].mean().to_dict()}")
    
    print(f"\nMacro summary (last 4 quarters):")
    print(macro_df.tail(4)[['date', 'gdp_growth', 'inflation', 'policy_rate', 'default_rate']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
