# Atelier FastAPI : déploiement Fashion-MNIST

API FastAPI qui reçoit une image de 784 pixels et renvoie la classe, la confiance et la latence d'inférence.

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload            # http://127.0.0.1:8000/docs
python test_api.py                  # 1 image, 20 images, entrée incorrecte
streamlit run streamlit_app.py      # front optionnel : http://localhost:8501
```

`test_api.py` charge Fashion-MNIST via Keras ; si le téléchargement échoue, il lit les fichiers du dossier `data/`.
Pour tester une API déployée : `API_URL=https://mon-api.exemple python test_api.py`.

## Contrat d'inférence

`POST /predict` avec `{"pixels": [784 nombres]}` (valeurs 0–1 ou 0–255) renvoie :

```json
{"class_id": 9, "class_name": "Ankle boot", "confidence": 0.9652, "latency_ms": 2.18}
```

Erreurs : `400` si le nombre de pixels est différent de 784, `422` si le JSON est mal formé.

## Déployer avec Docker

```bash
docker compose up -d --build        # API :8000, front Streamlit :8501
```

Sur une VM (Oracle Cloud par exemple) : copier le dossier, lancer la même commande, ouvrir les ports 8000/8501
(ou les mettre derrière un reverse proxy).

Sur Render / Railway / Hugging Face Spaces (SDK Docker) : pousser le dossier dans un dépôt GitHub et créer un
service web à partir du `Dockerfile`. Le conteneur écoute sur `$PORT` (8000 par défaut).
