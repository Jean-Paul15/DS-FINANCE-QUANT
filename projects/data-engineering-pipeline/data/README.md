# Données — EWS-Credit UMOA

## `raw/` — données réelles téléchargées (non versionnées, voir `.gitignore`)

Aucune microdonnée client de portefeuille de crédit UMOA n'existe en accès public
(secret bancaire, données de supervision BCEAO confidentielles). Ce qui suit est
donc la meilleure base réelle disponible, utilisée pour calibrer le générateur
synthétique plutôt que d'inventer des paramètres à l'aveugle.

| Dossier / fichier | Source | Contenu | Usage |
|---|---|---|---|
| `bceao_mgr_ratios.json` | API DBnomics/BCEAO, dataset MGR (sans authentification) | Taux moyen des crédits, marge, provisionnement — par pays UMOA, annuel | Calibrage taux d'intérêt et taux de dégradation |
| `bceao_iif_inclusion.json` | API DBnomics/BCEAO, dataset IIF | Encours de crédits, dépôts — indicateurs d'inclusion financière | Ordres de grandeur de volumes |
| `bceao_imf_bancarisation.json` | API DBnomics/BCEAO, dataset IMF | Taux de bancarisation par pays UMOA (2022) | Pondération de la répartition géographique du portefeuille |
| `home_credit_default_risk/` | Kaggle, competition *Home Credit Default Risk* | Panels mensuels réels (retards de paiement, soldes carte, historique bureau), ~2,7 Go | Calibrage des fréquences de retard mensuelles |
| `african_credit_scoring_zindi/` | Zindi *African Credit Scoring Challenge*, redistribution MIT sur Kaggle | 68 654 prêts réels, 6 540 clients kényans, prêts répétés, vrai défaut | Ancrage du taux de défaut cumulé |
| `taiwan_credit_default.xls` | UCI Machine Learning Repository | 30 000 clients, retards sur 6 mois | Référence académique |
| `german_credit_statlog.data` | UCI Machine Learning Repository (Statlog) | 1 000 clients, snapshot statique | Référence académique |

## `synthetic/` — portefeuille généré (`src/ews_credit`)

Généré par `python -m ews_credit.cli --n-clients 15000`. Quatre tables :

- **`clients.csv`** : démographie (pays UMOA pondéré par le vrai taux de
  bancarisation, segment particulier/PME, revenu, ancienneté).
- **`credits.csv`** : un ou deux crédits par client (type, montant, durée, taux
  d'intérêt centré sur 9,4 % — taux réel BCEAO 2020).
- **`panel_mensuel.csv`** : historique comportemental mensuel par crédit (DPD,
  Stage IFRS 9, solde restant dû, dépassement de découvert) — le cœur du
  dataset EWS. Simulé par une chaîne à états (Sain → petit retard → Stage 2 →
  Stage 3) calibrée sur les fréquences réelles Home Credit, avec un choc macro
  (6 mois) inspiré du vrai bond de provisionnement BCEAO 2019→2020 (x7).
- **`labels.csv`** : cible du modèle EWS — `bascule_defaut_3m` (1 si le crédit
  entre en Stage 3 dans les 3 mois suivants), calculée strictement depuis le
  futur du panel pour éviter toute fuite d'information.

Taux observés dans le portefeuille généré (15 000 clients) : taux de bascule à
3 mois ≈ 0,9 % (plus élevé pendant la période de choc simulée), taux de défaut
cumulé sur la vie du crédit ≈ 3,9 % — du bon ordre de grandeur pour un
portefeuille de détail, sans être un simple copier-coller d'une seule source
réelle.

Les constantes de calibration (avec leur source) sont dans
`src/ews_credit/config.py`.
