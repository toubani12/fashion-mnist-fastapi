"""Adapte une vraie photo de vêtement au format d'entrée du modèle Fashion-MNIST.

Les images Fashion-MNIST ont été fabriquées à partir de photos produit Zalando :
objet détouré, recadré, côté le plus long ramené à 28 px, centré sur un carré
28 × 28, niveaux de gris inversés (fond noir, objet clair). On reproduit ces
étapes pour qu'une photo prise au téléphone ressemble à ce que le modèle a vu
pendant l'entraînement.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # photos HEIC des iPhone
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

WORK_SIZE = 512  # les grandes photos sont réduites avant traitement


def _otsu_threshold(values):
    """Seuil d'Otsu sur des valeurs 0–255 (sépare objet et fond)."""
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    total = values.size
    cum_count = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    global_mean = cum_mean[-1] / total
    w0 = cum_count / total
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (global_mean * w0 - cum_mean / total) ** 2 / (w0 * w1)
    return int(np.nanargmax(between))


def _load(data):
    img = Image.open(BytesIO(data)) if isinstance(data, (bytes, bytearray)) else data
    img = ImageOps.exif_transpose(img)
    # fond transparent (PNG détouré) → fond blanc, comme une photo produit
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    img = img.convert("L")
    img.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def _orient_like_dataset(img, ratio=1.3, threshold=20, margin=1.1):
    mask = img > threshold
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(cols) < 4 or len(rows) < 2:
        return img
    width, height = cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1
    if width < ratio * height:  # objet debout (haut, pantalon, robe…) : on ne touche à rien
        return img
    heights = []
    for c in range(cols[0], cols[-1] + 1):
        r = np.nonzero(mask[:, c])[0]
        heights.append(r[-1] - r[0] + 1 if len(r) else 0)
    heights = np.array(heights, dtype=np.float32)
    k = max(2, len(heights) // 3)
    if heights[:k].mean() > margin * heights[-k:].mean():
        return img[:, ::-1].copy()
    return img


def to_fashion_mnist(data):
    """bytes ou PIL.Image → tableau uint8 (28, 28) au style Fashion-MNIST."""
    gray = _load(data)

    # 1. Fond : estimé sur le bord de l'image. Fond clair → on inverse.
    border = np.concatenate([gray[:4].ravel(), gray[-4:].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()])
    background = float(np.median(border))
    if background > 127:
        gray = 255.0 - gray
        background = 255.0 - background

    # 2. Le fond devient noir, l'objet garde ses nuances.
    img = np.clip(gray - background, 0, 255)
    img = np.asarray(Image.fromarray(img.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), dtype=np.float32)

    # 3. Masque de l'objet (Otsu) puis boîte englobante robuste aux poussières.
    threshold = max(_otsu_threshold(img), 12)
    mask = img > threshold
    if mask.sum() < 30:  # rien de détectable : on garde toute l'image
        ys, xs = np.arange(img.shape[0]), np.arange(img.shape[1])
    else:
        ys, xs = np.nonzero(mask)
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
    x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
    crop = img[y0:y1 + 1, x0:x1 + 1]

    # 4. Côté le plus long → 28 px, proportions conservées.
    h, w = crop.shape
    scale = 28.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # 5. Centrage sur un carré 28 × 28 noir.
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - new_w) // 2, (28 - new_h) // 2))
    out = np.asarray(canvas, dtype=np.float32)

    # 6. Orientation : dans Fashion-MNIST les chaussures ont toujours la pointe
    #    à gauche et le talon (ou la tige) à droite. Pour un objet couché, si le
    #    côté le plus haut est à gauche, on retourne l'image horizontalement.
    out = _orient_like_dataset(out)

    # 7. Contraste : la partie la plus claire de l'objet est ramenée à 255.
    peak = np.percentile(out[out > 0], 99) if (out > 0).any() else 0
    if peak > 0:
        out = np.clip(out * (255.0 / peak), 0, 255)
    return out.astype(np.uint8)
