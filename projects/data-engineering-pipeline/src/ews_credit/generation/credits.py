"""Generation des credits accordes a chaque client du portefeuille."""

import numpy as np
import pandas as pd

from ews_credit import config

MONTANT_MEDIAN_FCFA = {
    "Consommation": 800_000,
    "Habitat": 12_000_000,
    "Decouvert": 300_000,
    "Equipement PME": 6_000_000,
}
DUREE_MOIS_BORNES = {
    "Consommation": (6, 36),
    "Habitat": (60, 240),
    "Decouvert": (1, 12),
    "Equipement PME": (12, 60),
}


def _tirer_type_credit(rng: np.random.Generator, type_client: str) -> str:
    return rng.choice(config.TYPES_CREDIT, p=config.POIDS_TYPE_CREDIT[type_client])


def _tirer_taux_interet(rng: np.random.Generator, n: int) -> np.ndarray:
    taux = rng.normal(config.TAUX_INTERET_MOYEN_PCT, config.TAUX_INTERET_ECART_TYPE_PCT, size=n)
    lo, hi = config.TAUX_INTERET_BORNES_PCT
    return np.clip(taux, lo, hi).round(2)


def generer_credits(clients: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Genere un a deux credits par client, rattaches a client_id."""
    rng = np.random.default_rng(seed + 1)
    lignes = []
    credit_seq = 0

    for _, client in clients.iterrows():
        nb_credits = 1 if rng.random() < 0.78 else 2
        for _ in range(nb_credits):
            type_credit = _tirer_type_credit(rng, client["type_client"])
            mediane = MONTANT_MEDIAN_FCFA[type_credit]
            montant = rng.lognormal(mean=np.log(mediane), sigma=0.45)
            lo, hi = DUREE_MOIS_BORNES[type_credit]
            duree_mois = int(rng.integers(lo, hi + 1))
            debut_max_mois_avant = min(client["anciennete_bancaire_mois"], config.HORIZON_PANEL_MOIS)
            mois_octroi_relatif = int(rng.integers(0, max(debut_max_mois_avant, 1)))

            lignes.append(
                {
                    "credit_id": f"CRD{200000 + credit_seq}",
                    "client_id": client["client_id"],
                    "type_credit": type_credit,
                    "montant_initial_fcfa": round(montant, -3),
                    "duree_mois": duree_mois,
                    "taux_interet_annuel_pct": None,  # rempli en bloc ci-dessous
                    "mois_octroi_relatif": mois_octroi_relatif,
                }
            )
            credit_seq += 1

    credits = pd.DataFrame(lignes)
    credits["taux_interet_annuel_pct"] = _tirer_taux_interet(rng, len(credits))
    return credits
