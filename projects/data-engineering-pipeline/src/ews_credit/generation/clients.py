"""Generation de la population de clients du portefeuille synthetique."""

import numpy as np
import pandas as pd

from ews_credit import config


def _tirer_pays(rng: np.random.Generator, n: int) -> np.ndarray:
    poids = np.array([p.taux_bancarisation_pct for p in config.PAYS_UMOA])
    poids = poids / poids.sum()
    noms = [p.nom for p in config.PAYS_UMOA]
    return rng.choice(noms, size=n, p=poids)


def _tirer_revenu(rng: np.random.Generator, type_client: np.ndarray) -> np.ndarray:
    revenu = np.empty(len(type_client), dtype=float)
    for segment in config.TYPES_CLIENT:
        mask = type_client == segment
        mediane = config.REVENU_MENSUEL_MEDIAN_FCFA[segment]
        sigma = config.REVENU_MENSUEL_SIGMA_LOG[segment]
        revenu[mask] = rng.lognormal(mean=np.log(mediane), sigma=sigma, size=mask.sum())
    return revenu.round(-3)


def generer_clients(n_clients: int, seed: int = config.SEED) -> pd.DataFrame:
    """Genere la table clients : demographie et segmentation, sans historique."""
    rng = np.random.default_rng(seed)

    type_client = rng.choice(config.TYPES_CLIENT, size=n_clients, p=config.POIDS_TYPE_CLIENT)
    age = np.where(
        type_client == "Particulier",
        rng.integers(21, 68, size=n_clients),
        rng.integers(1, 25, size=n_clients),  # age = anciennete de l'entreprise en annees pour PME
    )
    anciennete_bancaire_mois = rng.integers(1, 180, size=n_clients)

    clients = pd.DataFrame(
        {
            "client_id": [f"CLI{100000 + i}" for i in range(n_clients)],
            "pays": _tirer_pays(rng, n_clients),
            "type_client": type_client,
            "age_ou_anciennete_entreprise_ans": age,
            "anciennete_bancaire_mois": anciennete_bancaire_mois,
            "revenu_mensuel_fcfa": _tirer_revenu(rng, type_client),
        }
    )

    # Score de risque latent : synthese non-observable des facteurs de fragilite
    # financiere du client (utilise par generation.behavior pour piloter les
    # probabilites de retard — jamais expose tel quel dans les sorties finales,
    # seul son effet sur le comportement observable doit etre appris par un modele).
    facteur_revenu = 1 - (clients["revenu_mensuel_fcfa"].rank(pct=True))
    facteur_anciennete = 1 - (clients["anciennete_bancaire_mois"].rank(pct=True))
    bruit_idiosyncratique = rng.normal(0, 0.15, size=n_clients)
    score_latent = 0.5 * facteur_revenu + 0.3 * facteur_anciennete + 0.2 * (bruit_idiosyncratique - bruit_idiosyncratique.min())
    clients["_score_risque_latent"] = (score_latent - score_latent.min()) / (score_latent.max() - score_latent.min())

    return clients
