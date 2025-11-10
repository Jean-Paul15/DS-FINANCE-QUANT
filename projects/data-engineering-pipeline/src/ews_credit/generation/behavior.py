"""Generation du panel comportemental mensuel — coeur du dataset EWS.

Simule, mois par mois et credit par credit, l'evolution du retard de
paiement (DPD) sous forme d'une chaine a etats (Sain / Watch / Sub-standard
/ Defaut) dont les probabilites de transition sont calibrees sur les
frequences reelles observees dans Home Credit Default Risk (cf. config.py)
et modulees par le score de risque latent du client et un choc macro.
"""

import numpy as np
import pandas as pd

from ews_credit import config
from ews_credit.domain.ifrs9 import stage_depuis_dpd, STAGE_DEFAUT

# Echelle de DPD utilisee comme etats discrets de la chaine de transition.
# "Petit retard" reste en Stage 1 (< SEUIL_DPD_STAGE_2) : la tres grande majorite
# des retards observes chez Home Credit (8.4% des echeances) se resorbent avant
# de devenir un vrai Stage 2, cf. proba_guerison_petit_retard ci-dessous.
DPD_PETIT_RETARD = 10
DPD_STAGE_2 = 45
DPD_STAGE_3 = 95

# Bornes volontairement resserrees : le multiplicateur agit successivement sur
# trois transitions (entree en retard, aggravation Stage 2, aggravation Stage 3),
# un facteur trop large composerait de facon irrealiste sur un client a risque.
MULTIPLICATEUR_RISQUE_BORNES = (0.3, 2.2)


def _multiplicateur_risque(score_latent: float) -> float:
    lo, hi = MULTIPLICATEUR_RISQUE_BORNES
    return lo + (hi - lo) * score_latent


def _multiplicateur_choc(mois_relatif: int) -> float:
    # Le facteur reel BCEAO (x7 sur le provisionnement agrege, cf. config) se
    # traduit dans le simulateur par trois transitions successives (entree en
    # retard, aggravation Stage 2, aggravation Stage 3) : on applique sa racine
    # cubique a chaque etape pour retrouver un choc final du bon ordre de
    # grandeur sans le composer artificiellement trois fois.
    if mois_relatif in config.MOIS_CHOC_MACRO:
        return config.INTENSITE_CHOC_MACRO ** (1 / 3)
    return 1.0


def _simuler_dpd_credit(rng: np.random.Generator, score_latent: float, n_mois: int) -> np.ndarray:
    dpd = np.zeros(n_mois, dtype=int)
    dpd_courant = 0
    mult_risque = _multiplicateur_risque(score_latent)

    for mois in range(n_mois):
        mult = mult_risque * _multiplicateur_choc(mois)

        if dpd_courant >= config.SEUIL_DPD_STAGE_3:
            dpd[mois] = dpd_courant  # Stage 3 : etat absorbant sur l'horizon du panel
            continue

        if dpd_courant == 0:
            proba_entree_retard = min(config.PROBA_BASE_SAIN_VERS_PETIT_RETARD * mult, 0.6)
            if rng.random() < proba_entree_retard:
                dpd_courant = DPD_PETIT_RETARD
        elif dpd_courant < config.SEUIL_DPD_STAGE_2:
            proba_guerison = max(min(config.PROBA_GUERISON_PETIT_RETARD / mult, 1.0), 0.0)
            if rng.random() < proba_guerison:
                dpd_courant = 0
            else:
                proba_aggravation = min(config.PROBA_BASE_PETIT_RETARD_VERS_STAGE2 * mult, 0.5)
                dpd_courant = DPD_STAGE_2 if rng.random() < proba_aggravation else DPD_PETIT_RETARD
        else:  # Stage 2 confirme (30-89j)
            proba_guerison = max(min(config.PROBA_GUERISON_STAGE2 / mult, 1.0), 0.0)
            if rng.random() < proba_guerison:
                dpd_courant = DPD_PETIT_RETARD
            else:
                proba_aggravation = min(config.PROBA_BASE_STAGE2_VERS_STAGE3 * mult, 0.5)
                dpd_courant = config.SEUIL_DPD_STAGE_3 if rng.random() < proba_aggravation else DPD_STAGE_2

        dpd[mois] = dpd_courant

    return dpd


def _solde_restant(montant_initial: float, duree_mois: int, mois_ecoules: int, dpd: int) -> float:
    amortissement_lineaire = max(0.0, 1 - mois_ecoules / duree_mois)
    if dpd >= config.SEUIL_DPD_STAGE_2:
        # capital + interets de retard qui continuent de courir : le solde ne baisse plus
        amortissement_lineaire = max(0.0, 1 - (mois_ecoules - dpd / 30) / duree_mois)
    return round(montant_initial * amortissement_lineaire, 0)


def generer_panel_mensuel(
    clients: pd.DataFrame, credits: pd.DataFrame, seed: int = config.SEED
) -> pd.DataFrame:
    """Genere le panel mensuel (une ligne par credit actif et par mois)."""
    rng = np.random.default_rng(seed + 2)
    score_par_client = clients.set_index("client_id")["_score_risque_latent"]

    lignes = []
    for _, credit in credits.iterrows():
        score_latent = score_par_client.loc[credit["client_id"]]
        mois_debut = int(credit["mois_octroi_relatif"])
        mois_fin = min(mois_debut + int(credit["duree_mois"]), config.HORIZON_PANEL_MOIS)
        n_mois = mois_fin - mois_debut
        if n_mois <= 0:
            continue

        dpd_serie = _simuler_dpd_credit(rng, score_latent, n_mois)
        est_decouvert = credit["type_credit"] == "Decouvert"

        for i, mois_relatif in enumerate(range(mois_debut, mois_fin)):
            dpd = int(dpd_serie[i])
            stage = stage_depuis_dpd(dpd)
            solde = _solde_restant(credit["montant_initial_fcfa"], credit["duree_mois"], i, dpd)
            depassement = 0.0
            if est_decouvert and rng.random() < 0.10 * _multiplicateur_risque(score_latent) / 6:
                depassement = round(solde * rng.uniform(0.05, 0.4), 0)

            lignes.append(
                {
                    "credit_id": credit["credit_id"],
                    "client_id": credit["client_id"],
                    "mois_relatif": mois_relatif,
                    "dpd_jours": dpd,
                    "stage_ifrs9": stage,
                    "solde_restant_du_fcfa": solde,
                    "depassement_decouvert_fcfa": depassement,
                }
            )

    panel = pd.DataFrame(lignes)
    panel["date_releve"] = pd.to_datetime(config.DATE_DEBUT_PANEL) + pd.to_timedelta(
        panel["mois_relatif"] * 30, unit="D"
    )
    return panel


def _bascule_a_venir_par_credit(stage_ifrs9: np.ndarray, horizon_mois: int) -> np.ndarray:
    """Pour chaque position i, vrai si Stage 3 apparait entre i+1 et i+horizon_mois inclus."""
    est_defaut = stage_ifrs9 == STAGE_DEFAUT
    n = len(est_defaut)
    label = np.zeros(n, dtype=int)
    for i in range(n):
        fenetre = est_defaut[i + 1 : i + 1 + horizon_mois]
        label[i] = int(fenetre.any())
    return label


def calculer_labels_bascule_defaut(panel: pd.DataFrame, horizon_mois: int = 3) -> pd.DataFrame:
    """Pour chaque (credit, mois) hors defaut, indique si le credit bascule en Stage 3
    dans les `horizon_mois` mois suivants — c'est la cible que le modele EWS doit predire.
    Calcule strictement a partir du futur du panel : aucune fuite d'information vers les
    variables explicatives disponibles a la date du releve.
    """
    panel_triee = panel.sort_values(["credit_id", "mois_relatif"]).reset_index(drop=True)
    nom_colonne = f"bascule_defaut_{horizon_mois}m"

    panel_triee[nom_colonne] = 0
    for _, indices in panel_triee.groupby("credit_id").groups.items():
        sous_serie = panel_triee.loc[indices, "stage_ifrs9"].to_numpy()
        panel_triee.loc[indices, nom_colonne] = _bascule_a_venir_par_credit(sous_serie, horizon_mois)

    colonnes = ["credit_id", "client_id", "mois_relatif", "date_releve", "stage_ifrs9", nom_colonne]
    labels = panel_triee.loc[panel_triee["stage_ifrs9"] != STAGE_DEFAUT, colonnes]
    return labels.reset_index(drop=True)
