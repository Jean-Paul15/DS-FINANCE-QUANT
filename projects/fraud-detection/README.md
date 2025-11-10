# Détection de fraude sur transactions

Scoring de fraude sur transactions de paiement mobile synthétiques, à très fort déséquilibre de classes, avec API de scoring et dashboard.

## Approche

1. Génération de comptes, transactions et graphe de transactions avec cinq types de fraude injectés (`src/data/`).
2. Features transactionnelles, comportementales (fenêtres glissantes) et de graphe (`src/features/`).
3. Split temporel avec contrôle de fuite, preprocessing et sélection de variables (`src/preprocessing/`).
4. Modèles : Isolation Forest, autoencodeur PyTorch, LightGBM (SMOTE, poids de classes), GraphSAGE optionnel (`torch_geometric`), puis stacking par régression logistique (`src/models/`).
5. Évaluation orientée fraude : AUC-PR, lift, recall à FPR fixé, analyse coût-bénéfice (`src/evaluation/fraud_metrics.py`).
6. Seuil adaptatif par bandit contextuel (`src/models/bandit.py`).

Les performances ne sont pas reportées ici : elles dépendent du générateur et doivent être reproduites avec `master_pipeline.py`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/data/generate_fraud_data.py   # écrit data/raw, data/processed, data/graph
python master_pipeline.py                # entraîne et écrit outputs/models/
```

Inférence :

```python
from inference import FraudDetector

detector = FraudDetector()
tx = {"transaction_id": "TX-1", "timestamp": "2024-01-15T14:30:00",
      "amount": 150000, "sender_id": "ACC-001", "receiver_id": "ACC-002",
      "transaction_type": "transfer", "channel": "mobile_app"}
score = detector.predict(tx)
rapport = detector.predict_full(tx)
```

API et dashboard :

```bash
uvicorn api.main:app --port 8001            # GET /health, POST /score, POST /score/batch
streamlit run dashboard/app.py --server.port 8501
docker compose up --build                   # monte ./outputs/models en lecture seule
```

## Tests

```bash
python -m pytest tests
```

`test_inference.py` requiert des modèles entraînés dans `outputs/models/`.

## Limites

Données synthétiques ; pipeline monolithique ; le GNN est optionnel et nécessite `torch_geometric`, absent de `requirements.txt`.
