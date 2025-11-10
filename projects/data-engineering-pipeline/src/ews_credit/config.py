"""Parametres de calibration du portefeuille synthetique EWS-Credit.

Toutes les constantes marquees [SOURCE REELLE] proviennent de donnees
publiques telechargees dans data/raw/ (API DBnomics/BCEAO, Kaggle Home
Credit Default Risk, Zindi African Credit Scoring Challenge). Elles ne
sont pas inventees : elles calibrent le generateur sur des ordres de
grandeur et des distributions observes dans la vraie vie, en l'absence
de microdonnees individuelles UMOA (confidentielles, non publiques).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PaysUMOA:
    nom: str
    code_iso: str
    taux_bancarisation_pct: float  # [SOURCE REELLE] BCEAO IMF, SF4003A0AP, 2022


# [SOURCE REELLE] BCEAO / DBnomics, dataset IMF (Indicateurs de la microfinance),
# serie SF4003A0AP "Taux de bancarisation strict (base population adulte)", derniere
# annee disponible = 2022. Utilise comme poids relatif de taille de portefeuille
# bancarise par pays (proxy defendable en l'absence de repartition officielle du
# nombre de comptes de credit par pays).
PAYS_UMOA = [
    PaysUMOA("Côte d'Ivoire", "CI", 29.508),
    PaysUMOA("Bénin", "BJ", 35.708),
    PaysUMOA("Burkina Faso", "BF", 21.754),
    PaysUMOA("Mali", "ML", 24.011),
    PaysUMOA("Niger", "NE", 8.678),
    PaysUMOA("Sénégal", "SN", 22.468),
    PaysUMOA("Guinée-Bissau", "GW", 16.398),
    PaysUMOA("Togo", "TG", 29.757),
]

TYPES_CLIENT = ["Particulier", "PME"]
POIDS_TYPE_CLIENT = [0.82, 0.18]  # portefeuille de detail majoritaire particuliers

TYPES_CREDIT = ["Consommation", "Habitat", "Decouvert", "Equipement PME"]
POIDS_TYPE_CREDIT = {
    "Particulier": [0.62, 0.18, 0.20, 0.0],
    "PME": [0.0, 0.0, 0.35, 0.65],
}

# [SOURCE REELLE] BCEAO / DBnomics, dataset MGR, serie SF2015A0AP "Taux moyen des
# credits a la clientele (en %)", ENSEMBLE UMOA, 2020 = 9.4 %. Ecart-type choisi
# pour etaler les taux entre types de credit (decouvert plus cher, habitat moins cher).
TAUX_INTERET_MOYEN_PCT = 9.4
TAUX_INTERET_ECART_TYPE_PCT = 2.1
TAUX_INTERET_BORNES_PCT = (6.0, 18.0)

# [SOURCE REELLE] BCEAO / DBnomics, dataset MGR, serie SF2033A0AP "Effort net de
# provisionnement (en %)", ENSEMBLE UMOA : 3.2 % en 2019 (regime normal),
# 22.6 % en 2020 (choc COVID). Utilises comme bornes du taux de degradation
# annuel du portefeuille (probabilite qu'un credit sain bascule en defaut).
TAUX_DEGRADATION_ANNUEL_REGIME_NORMAL = 0.032
TAUX_DEGRADATION_ANNUEL_REGIME_CHOC = 0.226

# [SOURCE REELLE] Kaggle Home Credit Default Risk, table installments_payments.csv
# (13,6M echeances reelles) : part des echeances mensuelles payees avec au moins
# un jour de retard, >30 jours, >90 jours. Ce sont des probabilites MARGINALES
# par echeance (pas des probabilites de transition), donc elles ne sont pas
# chainees telles quelles dans le simulateur (l'effet composerait de facon
# irrealiste) : elles bornent l'ordre de grandeur des probabilites de transition
# mensuelles de base retenues ci-dessous, pour un client de risque median.
PROBA_MENSUELLE_RETARD_LEGER = 0.084   # DPD > 0, mesure reelle Home Credit
PROBA_MENSUELLE_RETARD_30J = 0.0028    # DPD > 30, mesure reelle Home Credit
PROBA_MENSUELLE_RETARD_90J = 0.00094   # DPD > 90, mesure reelle Home Credit

# Probabilites de transition mensuelles retenues pour la chaine a etats du
# simulateur (Sain -> petit retard -> Stage 2 -> Stage 3), pour un client de
# risque median (multiplicateur = 1). Ordre de grandeur ancre sur les frequences
# reelles ci-dessus (entree en retard proche de 8.4%, aggravations rares),
# sans chainer les ratios bruts qui composeraient de facon irrealiste.
PROBA_BASE_SAIN_VERS_PETIT_RETARD = 0.05
PROBA_GUERISON_PETIT_RETARD = 0.55
PROBA_BASE_PETIT_RETARD_VERS_STAGE2 = 0.05
PROBA_GUERISON_STAGE2 = 0.30
PROBA_BASE_STAGE2_VERS_STAGE3 = 0.06

# [SOURCE REELLE] Zindi African Credit Scoring Challenge (via Kaggle, licence MIT) :
# taux de defaut observe sur 68 654 prets reels au Kenya = 1.83 %. Utilise comme
# ancrage du taux de defaut cumule cible sur la duree de vie d'un credit court terme,
# du meme ordre de grandeur que le microcredit ouest-africain.
TAUX_DEFAUT_CUMULE_ANCRAGE = 0.0183

# Seuils IFRS 9 (reglementaires, pas calibres sur donnees — ce sont les seuils
# standards du referentiel : Stage 1 sain, Stage 2 degrade, Stage 3 defaut).
SEUIL_DPD_STAGE_2 = 30
SEUIL_DPD_STAGE_3 = 90

REVENU_MENSUEL_MEDIAN_FCFA = {
    "Particulier": 180_000,
    "PME": 950_000,
}
REVENU_MENSUEL_SIGMA_LOG = {
    "Particulier": 0.55,
    "PME": 0.65,
}

HORIZON_PANEL_MOIS = 36
DATE_DEBUT_PANEL = "2022-01-01"

# Mois (index 0-based depuis DATE_DEBUT_PANEL) marquant un choc macro,
# calibre sur le vrai bond de provisionnement BCEAO 2019->2020 (x7).
MOIS_CHOC_MACRO = list(range(3, 9))  # 6 mois de choc, ex. avril-septembre an 1
INTENSITE_CHOC_MACRO = TAUX_DEGRADATION_ANNUEL_REGIME_CHOC / TAUX_DEGRADATION_ANNUEL_REGIME_NORMAL

SEED = 42
