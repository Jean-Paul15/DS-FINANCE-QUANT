# Génération de données de stress testing

Script unique produisant trois jeux de données synthétiques : macroéconomie UEMOA trimestrielle (2000-2024), portefeuille de prêts avec défauts liés aux variables macro, scénarios de stress.

```bash
python generate_stress_data.py
```

Sortie dans `data_stress/` (non versionné) : `raw/macro_uemoa.csv`, `raw/loan_portfolio.csv`, `scenarios/stress_scenarios.csv`. Dépendances : pandas, numpy.

Limite : le dépôt ne contient que le générateur, pas de modèle de stress testing.
