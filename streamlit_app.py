"""Mini front Streamlit pour l'API Fashion-MNIST (optionnel).

streamlit run streamlit_app.py   (API_URL=... pour viser une autre API)
"""
import gzip
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

LOCAL_API = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def api_is_up(url):
    try:
        return requests.get(f"{url}/health", timeout=2).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Démarrage de l'API FastAPI…")
def ensure_api():
    """Renvoie l'URL de l'API à utiliser.

    - API_URL défini : on utilise cette API (Docker, VM, autre serveur) ;
    - une API tourne déjà sur le port 8000 : on l'utilise ;
    - sinon (Streamlit Community Cloud) : on lance app.py avec uvicorn dans un
      thread, le front continue d'appeler la vraie route /predict.
    """
    if os.environ.get("API_URL"):
        return os.environ["API_URL"].rstrip("/")
    if api_is_up(LOCAL_API):
        return LOCAL_API

    import threading
    import time

    import uvicorn
    from app import app as fastapi_app

    server = uvicorn.Server(uvicorn.Config(fastapi_app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(120):
        if api_is_up(LOCAL_API):
            break
        time.sleep(0.5)
    return LOCAL_API


st.set_page_config(page_title="Fashion-MNIST", page_icon="👕", layout="centered")
st.title("Fashion-MNIST · classification")
API_URL = ensure_api()
st.caption(f"API : {API_URL}")


@st.cache_data
def load_test_set():
    with gzip.open(DATA_DIR / "t10k-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(DATA_DIR / "t10k-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return x, y


with st.sidebar:
    try:
        h = requests.get(f"{API_URL}/health", timeout=10).json()
        st.success(f"API {h['status']} · {h['model_name']}")
    except Exception as e:
        st.error(f"API injoignable : {e}")

source = st.radio("Image", ["Jeu de test Fashion-MNIST", "Mon image"], horizontal=True)
true_label = None
image = None

if source == "Jeu de test Fashion-MNIST":
    x, y = load_test_set()
    idx = st.number_input("Index (0 – 9999)", 0, len(x) - 1, 0)
    if st.button("Image au hasard"):
        idx = int(np.random.randint(len(x)))
    image, true_label = x[idx], CLASS_NAMES[int(y[idx])]
else:
    up = st.file_uploader("PNG/JPG d'un vêtement", type=["png", "jpg", "jpeg"])
    invert = st.checkbox("Inverser (fond clair → fond noir comme Fashion-MNIST)", True)
    if up:
        img = Image.open(up).convert("L").resize((28, 28))
        image = np.array(ImageOps.invert(img) if invert else img)

if image is not None:
    c1, c2 = st.columns([1, 1.4])
    c1.image(Image.fromarray(image.astype(np.uint8)).resize((196, 196), Image.NEAREST),
             caption=f"Réelle : {true_label}" if true_label else "Image envoyée")
    r = requests.post(f"{API_URL}/predict",
                      json={"pixels": image.reshape(-1).astype(float).tolist()}, timeout=30)
    if r.ok:
        res = r.json()
        c2.metric("Classe prédite", res["class_name"])
        c2.metric("Confiance", f"{res['confidence']:.2%}")
        c2.metric("Latence d'inférence", f"{res['latency_ms']} ms")
        if true_label:
            (c2.success if res["class_name"] == true_label else c2.warning)(
                "Correct" if res["class_name"] == true_label else "Incorrect")
        with st.expander("Réponse JSON"):
            st.json(res)
    else:
        st.error(r.json())
