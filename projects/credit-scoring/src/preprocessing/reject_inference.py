"""
Reject Inference — Modelisation des clients refuses.

Probleme : En production, la banque refuse certains clients.
Ces refus n'ont jamais ete observes (pas de label defaut).
Le modele est entraine uniquement sur les acceptes → biais de selection.

Solutions implementees :
1. Hard Cutoff Augmentation (HCA) — la plus simple, la plus defendable
2. Parcelling (extrapolation du taux de defaut par segment de risque)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def identify_rejected_clients(df, score_column=None, rejection_rate=0.30):
    """Simule une politique de rejet basee sur le score de credit.

    Dans une vraie banque, les rejets sont identifies par le statut
    de la demande. Ici, on les simule en prenant les clients avec
    le plus bas score bureau (proxy de rejet).

    Parameters
    ----------
    df : DataFrame
    score_column : str or None
        Colonne de score. Si None, cree un proxy via bureau_score.
    rejection_rate : float
        Taux de rejet simule (ex: 30% des demandes)

    Returns
    -------
    DataFrame avec colonne 'rejected'
    """
    df = df.copy()

    if score_column is None:
        # Proxy de score : bureau_score principalement
        proxy = (
            df["bureau_score"].fillna(620) / 850 * 50
            - df["debt_to_income"].fillna(0.3) * 30
            + df["age"].fillna(30) / 75 * 20
        )
        df["_proxy_score"] = proxy
        score_column = "_proxy_score"

    threshold = df[score_column].quantile(rejection_rate)
    df["rejected"] = df[score_column] < threshold

    return df, dict(threshold=threshold, rejection_rate=rejection_rate)


def hard_cutoff_augmentation(df_acceptes, rejection_rate=0.30, random_state=42):
    """HCA : Re-entraine le modele en incluant les rejetes comme defauts.

    Principe : Les clients rejetes sont consideres comme "defaut non observe".
    On les ajoute a l'echantillon d'entrainement avec un label defaut=1,
    puis on entraine un modele qui apprend a les distinguer.

    C'est la methode la plus conservative (prudente) — on suppose
    que tous les rejetes auraient fait defaut.

    Parameters
    ----------
    df_acceptes : DataFrame avec colonne 'default'
    rejection_rate : float
    random_state : int

    Returns
    -------
    DataFrame augmente
    """
    rng = np.random.RandomState(random_state)

    # Identifier les "rejetes" (clients avec les pires caracteristiques)
    df_aug = df_acceptes.copy()
    df_aug, _ = identify_rejected_clients(df_aug, rejection_rate=rejection_rate)

    # Hard cutoff : tous les rejetes = defaut
    mask_rejected = df_aug["rejected"]
    df_aug_hca = df_acceptes.copy()
    df_aug_hca["_was_rejected"] = mask_rejected.values

    # On garde tous les clients mais on cree un poids pour les rejetes
    # Les rejetes sont dupliques avec label=1 (defaut)
    rejected_rows = df_acceptes[mask_rejected.values].copy()
    rejected_rows["default"] = 1  # Hypothese conservative

    # Ajouter les rejetes comme defauts supplementaires
    df_augmented = pd.concat([df_acceptes, rejected_rows], ignore_index=True)

    return df_augmented


def parcelling_augmentation(df_acceptes, rejection_rate=0.30, random_state=42):
    """Parcelling : Extrapole le taux de defaut par segment de risque.

    Plus sophistique que HCA. Pour chaque decile de score, on observe
    le taux de defaut des acceptes et on l'extrapole aux rejetes
    en supposant que le taux continue sa tendance.

    Parameters
    ----------
    df_acceptes : DataFrame
    rejection_rate : float
    random_state : int

    Returns
    -------
    DataFrame avec colonne 'default_inferred' (probabiliste)
    """
    df_aug, info = identify_rejected_clients(df_acceptes, rejection_rate=rejection_rate)

    # Creer les deciles de score proxy
    df_aug["_decile"] = pd.qcut(df_aug["_proxy_score"], 10, labels=False, duplicates="drop")

    # Taux de defaut par decile (uniquement sur les acceptes)
    decile_rates = df_aug[~df_aug["rejected"]].groupby("_decile")["default"].mean()

    # Extrapoler : les rejetes sont dans les deciles les plus bas
    # On leur attribue le taux de defaut du decile + une prime de risque
    df_result = df_acceptes.copy()

    # Mapping du taux de defaut par decile
    decile_map = decile_rates.to_dict()
    df_result["_decile"] = df_aug["_decile"].values
    df_result["rejected"] = df_aug["rejected"].values

    # Pour les rejetes, on assigne le taux du decile * 1.5 (prime conservative)
    def get_inferred_pd(row):
        if row["rejected"]:
            base_rate = decile_map.get(row["_decile"], 0.5)
            return min(base_rate * 1.5, 0.95)  # Cap a 95%
        return row["default"]

    df_result["default_inferred"] = df_result.apply(get_inferred_pd, axis=1)

    return df_result


class RejectInferenceAugmenter:
    """Augmente le dataset avec l'inference de rejet.

    Utilise en preprocessing avant l'entrainement du modele final.
    """

    def __init__(self, method="hca", rejection_rate=0.30, random_state=42):
        self.method = method
        self.rejection_rate = rejection_rate
        self.random_state = random_state

    def augment(self, df):
        if self.method == "hca":
            return hard_cutoff_augmentation(
                df, self.rejection_rate, self.random_state
            )
        elif self.method == "parcelling":
            return parcelling_augmentation(
                df, self.rejection_rate, self.random_state
            )
        else:
            raise ValueError(f"Methode inconnue: {self.method}")

    def report(self):
        return {
            "method": self.method,
            "rejection_rate": self.rejection_rate,
            "description": (
                "Hard Cutoff Augmentation : les rejetes sont traites comme defauts"
                if self.method == "hca"
                else "Parcelling : extrapolation du taux de defaut par segment"
            ),
        }
