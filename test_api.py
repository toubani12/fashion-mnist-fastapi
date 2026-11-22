"""Tests de l'API Fashion-MNIST.

Lancer d'abord le serveur :  uvicorn app:app --reload
Puis :                       python test_api.py
(API_URL=https://... python test_api.py pour tester la version déployée)
"""
import gzip
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
PROJECT_DIR = Path(__file__).resolve().parent
CAPTURES = PROJECT_DIR / "captures"
CAPTURES.mkdir(exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_fashion_mnist_test():
    """Charge le jeu de test Fashion-MNIST (Keras, sinon fichiers idx locaux)."""
    try:
        import tensorflow as tf
        _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        return x_test, y_test
    except Exception:
        data_dir = Path(os.environ.get("FMNIST_DIR", PROJECT_DIR / "data"))
        with gzip.open(data_dir / "t10k-images-idx3-ubyte.gz") as f:
            x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(data_dir / "t10k-labels-idx1-ubyte.gz") as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        return x_test, y_test


def predict(image):
    payload = {"pixels": image.reshape(-1).astype(float).tolist()}  # 784 pixels
    response = requests.post(PREDICT_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def test_une_image(x_test, y_test, index=0):
    print("=" * 60)
    print(f"1) Test d'une image (index {index})")
    image, true_id = x_test[index], int(y_test[index])
    result = predict(image)
    print("Réponse de l'API :", json.dumps(result, ensure_ascii=False))
    print(f"Classe réelle  : {CLASS_NAMES[true_id]}")
    print(f"Classe prédite : {result['class_name']}")
    print("Résultat       :", "Correct" if result["class_id"] == true_id else "Incorrect")

    # affichage de l'image testée
    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3.2, 3.6))
    plt.imshow(image, cmap="gray")
    plt.title(f"Réelle : {CLASS_NAMES[true_id]}\nPrédite : {result['class_name']} "
              f"({result['confidence']:.2%})", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(CAPTURES / "image_testee.png", dpi=150)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()
    return result


def test_plusieurs_images(x_test, y_test, n=20):
    print("=" * 60)
    print(f"2) Test de {n} images")
    rows = []
    for i in range(n):
        r = predict(x_test[i])
        true_id = int(y_test[i])
        rows.append({
            "image": i,
            "classe_reelle": CLASS_NAMES[true_id],
            "classe_predite": r["class_name"],
            "confiance": r["confidence"],
            "latence_ms": r["latency_ms"],
            "resultat": "Correct" if r["class_id"] == true_id else "Incorrect",
        })

    print(f"{'Image':<6}{'Classe réelle':<14}{'Classe prédite':<15}"
          f"{'Confiance':>10}{'Latence':>11}  Résultat")
    for r in rows:
        print(f"{r['image']:<6}{r['classe_reelle']:<14}{r['classe_predite']:<15}"
              f"{r['confiance']:>10.2f}{r['latence_ms']:>8.2f} ms  {r['resultat']}")

    correct = sum(r["resultat"] == "Correct" for r in rows)
    latence_moy = sum(r["latence_ms"] for r in rows) / n
    print("-" * 60)
    print(f"Prédictions correctes : {correct}/{n} ({correct / n:.0%})")
    print(f"Latence moyenne       : {latence_moy:.2f} ms")
    return rows, correct, latence_moy


def test_entree_incorrecte():
    print("=" * 60)
    print("3) Test d'une entrée incorrecte (3 pixels)")
    response = requests.post(PREDICT_URL, json={"pixels": [0.0, 0.5, 1.0]}, timeout=30)
    print("Code HTTP :", response.status_code)
    print("Réponse   :", json.dumps(response.json(), ensure_ascii=False))
    assert response.status_code == 400
    return response.status_code, response.json()


if __name__ == "__main__":
    print("API :", API_URL)
    print("/health :", requests.get(f"{API_URL}/health", timeout=30).json())
    x_test, y_test = load_fashion_mnist_test()
    une = test_une_image(x_test, y_test, index=0)
    rows, correct, latence_moy = test_plusieurs_images(x_test, y_test, n=20)
    code, body = test_entree_incorrecte()

    with open(PROJECT_DIR / "resultats.json", "w", encoding="utf-8") as f:
        json.dump({"api": API_URL, "une_image": une, "tableau": rows,
                   "correctes": correct, "total": len(rows),
                   "latence_moyenne_ms": round(latence_moy, 2),
                   "entree_incorrecte": {"status_code": code, "body": body}},
                  f, ensure_ascii=False, indent=2)
