from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class ImageInput(BaseModel):
    pixels: list[float]


# Création de l'API
app = FastAPI(
    title="Fashion-MNIST API",
    description="API de classification des vêtements Fashion-MNIST",
    version="1.0",
)

# Chemin du dossier contenant app.py
PROJECT_DIR = Path(__file__).resolve().parent

# Chemin du modèle
MODEL_PATH = PROJECT_DIR / "fashion_mnist.keras"

print("Chemin du modèle :", MODEL_PATH)
print("Fichier trouvé :", MODEL_PATH.exists())

# Chargement du modèle (une seule fois, au démarrage du serveur)
model = tf.keras.models.load_model(MODEL_PATH)

print("Modèle chargé avec succès")
print("Forme d'entrée :", model.input_shape)
print("Forme de sortie :", model.output_shape)

# Appel à blanc : le premier appel construit le graphe et serait plus lent
model(np.zeros((1, 28, 28), dtype=np.float32), training=False)


@app.get("/")
def home():
    return {"message": "Bienvenue dans Fashion-MNIST API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_PATH.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }


@app.post(
    "/predict",
    responses={400: {"description": "L'entrée ne contient pas exactement 784 pixels"}},
)
def predict(data: ImageInput):
    # 1-2. la liste "pixels" doit contenir exactement 784 valeurs
    if len(data.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail="L'entrée doit contenir exactement 784 pixels.",
        )

    # 3. conversion en tableau NumPy float32
    image = np.array(data.pixels, dtype=np.float32)

    # 4. batch de forme (1, 28, 28)
    image = image.reshape(1, 28, 28)

    # 5. normalisation si les pixels sont dans [0, 255]
    if image.max() > 1.0:
        image = image / 255.0

    # 6 + 10. inférence avec training=False, latence mesurée autour de l'appel
    start = time.perf_counter()
    logits = model(image, training=False)
    latency_ms = (time.perf_counter() - start) * 1000

    # 7. logits -> probabilités
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]

    # 8-9. classe la plus probable et confiance
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])

    # 11. réponse JSON
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }
