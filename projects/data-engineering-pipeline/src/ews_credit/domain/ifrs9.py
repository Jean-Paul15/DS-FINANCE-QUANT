"""Regles metier IFRS 9 : classification d'un encours en fonction du retard.

Fonctions pures, sans dependance d'infrastructure ni de generation
aleatoire — c'est la seule source de verite sur la definition d'un stage.
"""

from ews_credit.config import SEUIL_DPD_STAGE_2, SEUIL_DPD_STAGE_3

STAGE_SAIN = 1
STAGE_DEGRADE = 2
STAGE_DEFAUT = 3


def stage_depuis_dpd(jours_retard: int) -> int:
    """Classe un encours en Stage IFRS 9 a partir de son nombre de jours de retard (DPD)."""
    if jours_retard >= SEUIL_DPD_STAGE_3:
        return STAGE_DEFAUT
    if jours_retard >= SEUIL_DPD_STAGE_2:
        return STAGE_DEGRADE
    return STAGE_SAIN


def est_basculement_vers_defaut(stage_precedent: int, stage_courant: int) -> bool:
    """Vrai si un encours vient d'entrer en Stage 3 ce mois-ci (evenement EWS a predire)."""
    return stage_precedent != STAGE_DEFAUT and stage_courant == STAGE_DEFAUT
