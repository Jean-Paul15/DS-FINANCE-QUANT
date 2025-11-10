"""Orchestration : assemble clients, credits, panel comportemental et labels."""

from dataclasses import dataclass

import pandas as pd

from ews_credit import config
from ews_credit.generation.behavior import calculer_labels_bascule_defaut, generer_panel_mensuel
from ews_credit.generation.clients import generer_clients
from ews_credit.generation.credits import generer_credits


@dataclass(frozen=True)
class Portefeuille:
    clients: pd.DataFrame
    credits: pd.DataFrame
    panel_mensuel: pd.DataFrame
    labels: pd.DataFrame


def generer_portefeuille(n_clients: int, seed: int = config.SEED) -> Portefeuille:
    clients = generer_clients(n_clients, seed=seed)
    credits = generer_credits(clients, seed=seed)
    panel = generer_panel_mensuel(clients, credits, seed=seed)
    labels = calculer_labels_bascule_defaut(panel, horizon_mois=3)

    clients_exposes = clients.drop(columns=["_score_risque_latent"])
    return Portefeuille(clients=clients_exposes, credits=credits, panel_mensuel=panel, labels=labels)
