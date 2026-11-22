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

## Interface « Maison MNIST » (Streamlit)

- **Photo réelle** : import (JPG, PNG, WEBP, HEIC) ou appareil photo. La route `POST /predict/image` adapte
  automatiquement la photo au format du modèle (`preprocessing.py`) : fond détecté et inversé si clair,
  objet détouré et recadré, côté le plus long ramené à 28 px, centrage, chaussures orientées pointe à gauche
  comme dans Fashion-MNIST, contraste. La réponse contient aussi le top 3 et l'image 28 × 28 vue par le modèle.
- **Collection** : dix pièces du jeu de test, une par famille, à essayer en un clic.
- Conseil pour de bons résultats : une seule pièce, entière, sur fond uni.

## Contrat d'inférence

`POST /predict` avec `{"pixels": [784 nombres]}` (valeurs 0–1 ou 0–255) renvoie :

```json
{"class_id": 9, "class_name": "Ankle boot", "confidence": 0.9652, "latency_ms": 2.18}
```

Erreurs : `400` si le nombre de pixels est différent de 784, `422` si le JSON est mal formé.

Extensions : `POST /predict?top_k=3` ajoute les 3 classes les plus probables ; `POST /predict/image`
(multipart, champ `file`) accepte une vraie photo.

## Déployer sur Streamlit Community Cloud

1. share.streamlit.io → **Create app** → choisir ce dépôt, branche `main`, fichier `streamlit_app.py`.
2. Dans *Advanced settings*, choisir Python 3.12.
3. Déployer.

Streamlit Cloud ne lance que l'app Streamlit. Si aucune API n'est joignable (ni `API_URL`, ni port 8000),
`streamlit_app.py` charge l'application FastAPI de `app.py` dans son propre process et l'appelle avec
`fastapi.testclient.TestClient` : les requêtes passent toujours par `GET /health` et `POST /predict`
(validation Pydantic, erreur 400…), sans serveur ni port réseau. Pour viser une API hébergée ailleurs,
définir le secret `API_URL`.

## Déployer avec Docker

```bash
docker compose up -d --build        # API :8000, front Streamlit :8501
```

Sur une VM (Oracle Cloud par exemple) : copier le dossier, lancer la même commande, ouvrir les ports 8000/8501
(ou les mettre derrière un reverse proxy).

Sur Render / Railway / Hugging Face Spaces (SDK Docker) : pousser le dossier dans un dépôt GitHub et créer un
service web à partir du `Dockerfile`. Le conteneur écoute sur `$PORT` (8000 par défaut).
