from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from preprocessing import to_fashion_mnist

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


def run_inference(image):
    """image : tableau (28, 28) float32, pixels déjà dans [0, 1]."""
    batch = image.reshape(1, 28, 28)
    start = time.perf_counter()
    logits = model(batch, training=False)
    latency_ms = (time.perf_counter() - start) * 1000
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    return probabilities, latency_ms


def format_result(probabilities, latency_ms, top_k=None):
    class_id = int(np.argmax(probabilities))
    result = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
        "latency_ms": round(latency_ms, 2),
    }
    if top_k:
        order = np.argsort(probabilities)[::-1][:top_k]
        result["top_k"] = [
            {"class_id": int(i), "class_name": CLASS_NAMES[i], "probability": round(float(probabilities[i]), 4)}
            for i in order
        ]
    return result


@app.post(
    "/predict",
    responses={400: {"description": "L'entrée ne contient pas exactement 784 pixels"}},
)
def predict(data: ImageInput, top_k: int | None = Query(None, ge=1, le=10, description="Optionnel : renvoie aussi les k classes les plus probables")):
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

    # 6 à 10. inférence (training=False), softmax, classe, confiance, latence
    probabilities, latency_ms = run_inference(image)

    # 11. réponse JSON
    return format_result(probabilities, latency_ms, top_k)


@app.post(
    "/predict/image",
    responses={400: {"description": "Fichier illisible ou qui n'est pas une image"}},
)
async def predict_image(file: UploadFile = File(..., description="Photo d'un vêtement (JPG, PNG, WEBP…)")):
    """Extension : reçoit une vraie photo, l'adapte au format Fashion-MNIST
    (preprocessing.py) puis applique la même inférence que /predict."""
    data = await file.read()
    try:
        pixels = to_fashion_mnist(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image lisible.")
    probabilities, latency_ms = run_inference(pixels.astype(np.float32) / 255.0)
    result = format_result(probabilities, latency_ms, top_k=3)
    result["preview"] = pixels.reshape(-1).tolist()  # l'image 28×28 réellement vue par le modèle
    return result
