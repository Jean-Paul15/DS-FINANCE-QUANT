"""Point d'entree : genere le portefeuille synthetique et l'ecrit dans data/synthetic/."""

import argparse
from pathlib import Path

from ews_credit import config
from ews_credit.generation.portfolio import generer_portefeuille

DOSSIER_SORTIE_PAR_DEFAUT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere le portefeuille synthetique EWS-Credit UMOA.")
    parser.add_argument("--n-clients", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--sortie", type=Path, default=DOSSIER_SORTIE_PAR_DEFAUT)
    args = parser.parse_args()

    args.sortie.mkdir(parents=True, exist_ok=True)

    portefeuille = generer_portefeuille(args.n_clients, seed=args.seed)
    portefeuille.clients.to_csv(args.sortie / "clients.csv", index=False)
    portefeuille.credits.to_csv(args.sortie / "credits.csv", index=False)
    portefeuille.panel_mensuel.to_csv(args.sortie / "panel_mensuel.csv", index=False)
    portefeuille.labels.to_csv(args.sortie / "labels.csv", index=False)

    taux_bascule = portefeuille.labels["bascule_defaut_3m"].mean()
    print(f"Clients          : {len(portefeuille.clients):,}")
    print(f"Credits          : {len(portefeuille.credits):,}")
    print(f"Lignes panel     : {len(portefeuille.panel_mensuel):,}")
    print(f"Lignes labels    : {len(portefeuille.labels):,}")
    print(f"Taux bascule 3m  : {taux_bascule:.2%}")
    print(f"Sortie ecrite dans : {args.sortie}")


if __name__ == "__main__":
    main()
