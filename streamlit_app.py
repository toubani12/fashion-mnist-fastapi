"""Maison MNIST — interface Streamlit de l'API Fashion-MNIST.

streamlit run streamlit_app.py      (API_URL=... pour viser une API distante)
"""
import base64
import gzip
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

import preprocessing  # noqa: F401  (active la lecture des photos HEIC)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOCAL_API = "http://127.0.0.1:8000"

LABELS_FR = {
    "T-shirt/top": "T-shirt", "Trouser": "Pantalon", "Pullover": "Pull", "Dress": "Robe",
    "Coat": "Manteau", "Sandal": "Sandale", "Shirt": "Chemise", "Sneaker": "Basket",
    "Bag": "Sac", "Ankle boot": "Bottine",
}
CLASS_NAMES = list(LABELS_FR)
# une pièce nette par famille, tirée du jeu de test
GALLERY = [28, 18, 13, 6, 9, 1, 2, 82, 7, 19]

st.set_page_config(page_title="Maison MNIST · Reconnaissance de vêtements", page_icon="🖤", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Inter:wght@300;400;500;600&display=swap');
:root{--ink:#efe8f1;--muted:#a597a9;--faint:#6f6273;--lilac:#cbb2d6;--rose:#e3b7c8;--line:rgba(255,255,255,.08);}
html, body, [class*="css"], .stMarkdown, p, label, button, input {font-family:'Inter',sans-serif;}
[data-testid="stAppViewContainer"]{background:
  radial-gradient(1100px 600px at 8% -10%, rgba(110,52,118,.35), transparent 60%),
  radial-gradient(900px 600px at 100% 30%, rgba(70,34,84,.28), transparent 60%),
  #0d0a0f;}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer, [data-testid="stSidebar"], [data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:1180px;padding-top:1.6rem;padding-bottom:3rem;}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:3.2rem;}
.pill{display:inline-flex;align-items:center;gap:.55rem;padding:.42rem .9rem;border:1px solid var(--line);border-radius:999px;
  font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink);background:rgba(255,255,255,.03);}
.dot{width:7px;height:7px;border-radius:50%;background:#8fd6a5;box-shadow:0 0 10px #8fd6a5;}
.brand{font-family:'Cormorant Garamond',serif;font-size:1.55rem;color:var(--ink);letter-spacing:.02em;}
.brand b{font-weight:600;font-style:italic;color:var(--lilac);}
.kicker{font-size:.7rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin-bottom:.9rem;}
.hero h1{font-family:'Cormorant Garamond',serif !important;font-weight:500;font-size:4.1rem;line-height:1.02;color:var(--ink);margin:0 0 1rem;padding:0;}
.hero h1 em, .section-title em, .banner h2 em{color:var(--lilac);font-style:italic;}
.hero p{color:var(--muted);font-size:1.02rem;max-width:560px;line-height:1.65;font-weight:300;}
.card{background:linear-gradient(180deg,rgba(32,23,37,.92),rgba(19,14,22,.92));border:1px solid var(--line);border-radius:22px;padding:1.6rem 1.7rem;}
.step{font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;color:var(--faint);margin-bottom:.8rem;}
.step b{color:var(--lilac);font-weight:500;}
.tip{color:var(--faint);font-size:.8rem;line-height:1.55;margin-top:.6rem;}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.025);border:1px dashed rgba(203,178,214,.35);border-radius:18px;padding:2.1rem 1rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--lilac);background:rgba(203,178,214,.06);}
[data-testid="stFileUploaderDropzoneInstructions"] small{display:none;}
[data-testid="stCameraInput"] > div{border-radius:18px;overflow:hidden;}
.stButton>button{width:100%;background:transparent;border:1px solid rgba(255,255,255,.16);
  border-radius:999px;color:var(--ink);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;padding:.5rem 1rem;transition:all .25s ease;}
.stButton>button:hover{border-color:var(--lilac);background:rgba(203,178,214,.10);color:#fff;transform:translateY(-1px);}
.empty{min-height:430px;display:flex;flex-direction:column;justify-content:center;align-items:flex-start;}
.empty h3{font-family:'Cormorant Garamond',serif !important;font-weight:500;font-size:2.4rem;color:var(--ink);margin:.4rem 0 .6rem;padding:0;}
.empty p{color:var(--muted);font-weight:300;max-width:360px;line-height:1.6;}
.result{animation:rise .55s cubic-bezier(.2,.7,.2,1);}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.shots{display:flex;gap:14px;align-items:flex-end;margin-bottom:1.4rem;}
.shot-main{flex:1;height:250px;border-radius:18px;background:#0a080c;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;overflow:hidden;}
.shot-main img{max-width:100%;max-height:100%;object-fit:contain;}
.shot-model{width:118px;text-align:center;}
.shot-model img{width:118px;height:118px;border-radius:14px;image-rendering:pixelated;border:1px solid var(--line);}
.shot-model span{display:block;font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin-top:.5rem;line-height:1.4;}
.name{font-family:'Cormorant Garamond',serif;font-size:3.6rem;line-height:1;color:var(--ink);font-weight:500;}
.name-en{font-size:.7rem;letter-spacing:.22em;text-transform:uppercase;color:var(--faint);margin:.45rem 0 1.1rem;}
.verdict{display:inline-flex;gap:.5rem;align-items:center;padding:.35rem .85rem;border-radius:999px;font-size:.8rem;
  background:rgba(203,178,214,.10);border:1px solid rgba(203,178,214,.25);color:var(--lilac);}
.verdict.low{background:rgba(227,183,200,.08);border-color:rgba(227,183,200,.3);color:var(--rose);}
.bar{height:3px;border-radius:3px;background:rgba(255,255,255,.07);margin:1rem 0 1.4rem;overflow:hidden;}
.bar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,#8e6aa0,var(--lilac),var(--rose));}
.alts-title{font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--faint);margin-bottom:.55rem;}
.alt{display:grid;grid-template-columns:110px 1fr 48px;gap:.8rem;align-items:center;color:var(--muted);font-size:.86rem;margin:.35rem 0;}
.alt .mini{height:2px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden;}
.alt .mini i{display:block;height:100%;background:rgba(203,178,214,.6);}
.alt b{font-weight:400;text-align:right;color:var(--faint);}
.truth{margin-top:1.2rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.82rem;color:var(--muted);}
.truth .ok{color:#8fd6a5}.truth .ko{color:var(--rose)}
.speed{float:right;color:var(--faint);letter-spacing:.08em;}
.section-head{display:flex;justify-content:space-between;align-items:flex-end;margin:4.2rem 0 1.4rem;}
.section-title{font-family:'Cormorant Garamond',serif !important;font-size:2.6rem;font-weight:500;color:var(--ink);margin:0;line-height:1;padding:0;}
.section-head > span{color:var(--faint);font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;}
.piece{background:linear-gradient(180deg,rgba(32,23,37,.9),rgba(17,13,20,.9));border:1px solid var(--line);border-radius:18px;
  padding:.9rem .9rem 1rem;transition:transform .3s ease,border-color .3s ease;margin-bottom:.55rem;}
.piece:hover{transform:translateY(-3px);border-color:rgba(203,178,214,.35);}
.piece .no{font-size:.6rem;letter-spacing:.2em;color:var(--faint);}
.piece img{width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:12px;background:#0a080c;margin:.6rem 0 .7rem;}
.piece .pn{font-family:'Cormorant Garamond',serif;font-size:1.35rem;color:var(--ink);line-height:1;}
.banner{margin-top:4rem;display:grid;grid-template-columns:1.1fr 1fr;gap:2rem;align-items:center;}
.banner h2{font-family:'Cormorant Garamond',serif !important;font-size:3.2rem;font-weight:500;line-height:1.02;color:var(--ink);margin:0 0 1rem;padding:0;}
.banner p{color:var(--muted);font-weight:300;line-height:1.7;}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;}
.chips span{padding:.45rem .95rem;border-radius:999px;border:1px solid var(--line);color:var(--ink);font-size:.82rem;background:rgba(255,255,255,.025);}
.foot{margin-top:3.5rem;padding-top:1.2rem;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:1rem;color:var(--faint);font-size:.72rem;letter-spacing:.06em;}
div[role="dialog"]{background:linear-gradient(180deg,#1d1522,#120d15) !important;border:1px solid var(--line) !important;border-radius:24px !important;}
div[role="dialog"] .card{background:transparent !important;border:none !important;padding:0 !important;}
div[role="dialog"] h2{font-family:'Cormorant Garamond',serif !important;font-weight:500 !important;font-size:1.9rem !important;color:var(--ink);}
@media (max-width: 800px){.hero h1{font-size:2.8rem}.banner{grid-template-columns:1fr}.name{font-size:2.8rem}}
</style>
""", unsafe_allow_html=True)


def html(markup):
    """Affiche du HTML sans que Markdown n'interprète l'indentation."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def data_uri(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, fmt, quality=88)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


# --------------------------------------------------------------------------- API
def api_is_up(url):
    try:
        return requests.get(f"{url}/health", timeout=2).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Préparation du modèle…")
def get_api():
    """(libellé, fonction d'appel) vers l'API.

    API_URL défini ou API sur le port 8000 → appels HTTP. Sinon (Streamlit
    Community Cloud) l'application FastAPI de app.py est chargée ici et appelée
    via TestClient : mêmes routes, même validation, sans serveur réseau.
    """
    url = os.environ.get("API_URL") or (LOCAL_API if api_is_up(LOCAL_API) else None)
    if url:
        url = url.rstrip("/")
        session = requests.Session()
        return url, lambda method, path, **kw: session.request(method, url + path, timeout=60, **kw)
    from fastapi.testclient import TestClient
    from app import app as fastapi_app
    client = TestClient(fastapi_app)
    return "intégrée", lambda method, path, **kw: client.request(method, path, **kw)


API_LABEL, call_api = get_api()


@st.cache_data(show_spinner=False)
def load_test_set():
    with gzip.open(DATA_DIR / "t10k-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(DATA_DIR / "t10k-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return x, y


@st.cache_data(show_spinner=False, max_entries=64)
def predict_photo(data: bytes, name: str, mime: str):
    r = call_api("POST", "/predict/image", files={"file": (name, data, mime or "image/jpeg")})
    return r.status_code, r.json()


@st.cache_data(show_spinner=False)
def predict_sample(idx: int):
    x, _ = load_test_set()
    r = call_api("POST", "/predict?top_k=3", json={"pixels": x[idx].reshape(-1).astype(float).tolist()})
    return r.status_code, r.json()


def pretty_piece(arr28, size=220):
    """Vignette douce d'une image 28×28 pour la galerie."""
    return Image.fromarray(arr28.astype(np.uint8)).resize((size, size), Image.LANCZOS)


def open_photo(data):
    try:
        return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------- état
ss = st.session_state
ss.setdefault("active", None)       # "upload" | "camera"


def on_upload():
    ss.active = "upload" if ss.get("upload") is not None else None


def on_camera():
    ss.active = "camera" if ss.get("camera") is not None else None


def pick(idx):
    ss.dialog_idx = idx


# --------------------------------------------------------------------------- en-tête
html("""
<div class="topbar">
  <span class="pill"><span class="dot"></span>Atelier FastAPI · Fashion-MNIST</span>
  <span class="brand">maison <b>mnist</b></span>
</div>
<div class="hero">
  <div class="kicker">Reconnaissance de vêtements · 10 familles de pièces</div>
  <h1>Montrez une pièce.<br><em>On la reconnaît.</em></h1>
  <p>Importez la photo d'un vêtement ou d'un accessoire, ou prenez-la sur le moment.
  L'image est préparée automatiquement, puis le modèle vous dit de quelle pièce il s'agit.</p>
</div>
""")
st.write("")

left, right = st.columns([1, 1.18], gap="large")

with left:
    html('<div class="step"><b>01</b> — Votre pièce</div>')
    mode = st.segmented_control("Source", ["Importer une photo", "Prendre une photo"], default="Importer une photo",
                                label_visibility="collapsed", key="mode")
    if mode == "Prendre une photo":
        st.camera_input("Photographiez la pièce", key="camera", on_change=on_camera, label_visibility="collapsed")
    else:
        st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp", "heic"], key="upload", on_change=on_upload,
                         label_visibility="collapsed")
    html("""<div class="tip">Pour une lecture fiable : une seule pièce (une chaussure plutôt que la paire), entière et bien cadrée,
    posée sur un fond uni : mur, drap, sol clair ou sombre. Les photos de catalogue fonctionnent particulièrement bien.</div>""")

# --------------------------------------------------------------------------- résultat
def fr_pct(p, digits=1):
    return f"{p * 100:.{digits}f}".replace(".", ",") + " %"


def result_card(res, original, truth=None):
    conf = res["confidence"]
    if conf >= 0.9:
        verdict, cls = "Sans hésitation", ""
    elif conf >= 0.7:
        verdict, cls = "Très probablement", ""
    elif conf >= 0.5:
        verdict, cls = "Probablement", "low"
    else:
        verdict, cls = "Le modèle hésite", "low"
    thumb = original.copy()
    thumb.thumbnail((700, 700))
    preview = Image.fromarray(np.array(res["preview"], dtype=np.uint8).reshape(28, 28)).resize((112, 112), Image.NEAREST)
    alts = "".join(
        f'<div class="alt"><span>{LABELS_FR[a["class_name"]]}</span><div class="mini"><i style="width:{a["probability"] * 100:.1f}%"></i></div>'
        f'<b>{fr_pct(a["probability"], 0)}</b></div>'
        for a in res.get("top_k", [])[1:3]
    )
    truth_html = ""
    if truth:
        ok = truth == res["class_name"]
        truth_html = (f'<div class="truth">Pièce de la collection : <b>{LABELS_FR[truth]}</b> — '
                      f'<span class="{"ok" if ok else "ko"}">{"bonne réponse" if ok else "le modèle se trompe"}</span></div>')
    html(f"""
    <div class="card result">
      <div class="step"><b>02</b> — Notre lecture <span class="speed">analysée en {max(res['latency_ms'], 1):.0f} ms</span></div>
      <div class="shots">
        <div class="shot-main"><img src="{data_uri(thumb, 'JPEG')}"></div>
        <div class="shot-model"><img src="{data_uri(preview)}"><span>Ce que voit<br>le modèle</span></div>
      </div>
      <div class="name">{LABELS_FR[res['class_name']]}</div>
      <div class="name-en">{res['class_name']}</div>
      <span class="verdict {cls}">{verdict} · {fr_pct(conf)}</span>
      <div class="bar"><i style="width:{conf * 100:.1f}%"></i></div>
      <div class="alts-title">Autres pistes</div>
      {alts}
      {truth_html}
    </div>""")


@st.dialog("Pièce de la collection", width="medium")
def sample_dialog(idx):
    x, y = load_test_set()
    status, res = predict_sample(idx)
    if status != 200:
        st.error("Le service ne répond pas, réessayez dans un instant.")
        return
    res = dict(res, preview=x[idx].reshape(-1).tolist())
    result_card(res, pretty_piece(x[idx], 360).convert("RGB"), CLASS_NAMES[int(y[idx])])


status, res, original = None, None, None
if ss.active == "upload" and ss.get("upload") is not None:
    f = ss.upload
    data = f.getvalue()
    status, res = predict_photo(data, f.name, f.type)
    original = open_photo(data)
elif ss.active == "camera" and ss.get("camera") is not None:
    data = ss.camera.getvalue()
    status, res = predict_photo(data, "camera.jpg", "image/jpeg")
    original = open_photo(data)

with right:
    if res is None:
        html("""
        <div class="card empty">
          <div class="step"><b>02</b> — Notre lecture</div>
          <h3>Votre pièce apparaîtra ici.</h3>
          <p>Importez une photo à gauche, ou choisissez une pièce de la collection juste en dessous pour voir le modèle à l'œuvre.</p>
        </div>""")
    elif status != 200 or original is None:
        detail = res.get("detail") if isinstance(res.get("detail"), str) else "Essayez avec une autre photo (JPG ou PNG)."
        html(f"""<div class="card empty"><div class="step"><b>02</b> — Notre lecture</div>
        <h3>Image illisible.</h3><p>{detail}</p></div>""")
    else:
        result_card(res, original)

if ss.get("dialog_idx") is not None:
    sample_dialog(ss.pop("dialog_idx"))

# --------------------------------------------------------------------------- galerie
html("""
<div class="section-head">
  <div><div class="kicker">Ou essayez avec</div><h2 class="section-title">Les pièces de la <em>collection</em></h2></div>
  <span>10 familles · jeu de test</span>
</div>""")
x, y = load_test_set()
for row in (GALLERY[:5], GALLERY[5:]):
    cols = st.columns(5, gap="medium")
    for col, idx in zip(cols, row):
        with col:
            html(f"""
            <div class="piece">
              <div class="no">N°{GALLERY.index(idx) + 1:02d}</div>
              <img src="{data_uri(pretty_piece(x[idx]))}">
              <div class="pn">{LABELS_FR[CLASS_NAMES[int(y[idx])]]}</div>
            </div>""")
            st.button("Essayer", key=f"try{idx}", on_click=pick, args=(idx,), width="stretch")

chips = "".join(f"<span>{v}</span>" for v in LABELS_FR.values())
html(f"""
<div class="banner">
  <div class="card" style="padding:2.4rem 2.4rem">
    <div class="kicker">Comment ça marche</div>
    <h2>Une photo.<br><em>Une réponse.</em></h2>
    <p>Votre photo est détourée, recadrée, orientée, mise en niveaux de gris et réduite à 28 × 28 points, exactement comme les images
    sur lesquelles le modèle a appris. Il ne reste qu'à la lui montrer : il répond en quelques millisecondes.</p>
  </div>
  <div>
    <div class="kicker">Les dix familles reconnues</div>
    <div class="chips">{chips}</div>
  </div>
</div>
<div class="foot">
  <span>Atelier FastAPI · Master SDIA — ENSET Mohammedia · Badr Eddine Toubani</span>
  <span><span class="dot" style="display:inline-block;margin-right:.45rem"></span>Service en ligne · API {API_LABEL}</span>
</div>
""")
