# Pipeline de portefeuille de crédit synthétique (EWS)

Génération d'un portefeuille de crédit (clients, crédits, panel mensuel de retards) et de labels de bascule en défaut à 3 mois, pour alimenter un modèle d'alerte précoce.

## Contenu

- `src/ews_credit/config.py` : paramètres de calibrage (pays UMOA, taux, probabilités de transition).
- `src/ews_credit/domain/ifrs9.py` : classification en stage IFRS 9 à partir des jours de retard (fonctions pures).
- `src/ews_credit/generation/` : clients, crédits, panel mensuel (chaîne à états Sain / retard / Stage 2 / Stage 3 avec choc macro), labels calculés sur le futur du panel.
- `src/ews_credit/cli.py` : point d'entrée.

## Usage

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m ews_credit.cli --n-clients 15000 --seed 42
```

Écrit `clients.csv`, `credits.csv`, `panel_mensuel.csv` et `labels.csv` dans `data/synthetic/` (non versionné ; option `--sortie` pour changer le dossier).

## Données de calibrage

Voir [`data/README.md`](data/README.md). Les sources publiques (BCEAO via DBnomics, Home Credit, Zindi, UCI) ne sont pas redistribuées ; il faut les télécharger auprès de leurs éditeurs.

## Limites

Pas de tests automatisés. L'exécution complète du CLI n'a pas pu être validée dans l'environnement de nettoyage (arrêt brutal de l'interpréteur Python 3.14) ; la syntaxe a été vérifiée par compilation.
