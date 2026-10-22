"""OCR con Tesseract: lee las palabras escritas en páginas escaneadas y dentro de las imágenes.
Es MUY rápido en CPU (1-3 s por página) comparado con un modelo de visión (30 s - 2 min)."""

import io
import os
import shutil
from functools import lru_cache

from PIL import Image, ImageOps

from .config import BASE_DIR

# Si existe la carpeta "tessdata" del proyecto (iniciar.bat descarga ahí el español "rápido"), se usa esa.
LOCAL_TESSDATA = BASE_DIR / "tessdata"

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


@lru_cache(maxsize=4)
def find_tesseract(custom: str = "") -> str | None:
    for cand in [custom, shutil.which("tesseract"), *WINDOWS_PATHS]:
        if cand and (os.path.isfile(cand) or shutil.which(cand)):
            return cand
    return None


def _config() -> str:
    """Usa la carpeta tessdata del proyecto (vía TESSDATA_PREFIX: funciona aunque la ruta tenga espacios)."""
    if LOCAL_TESSDATA.is_dir() and any(LOCAL_TESSDATA.glob("*.traineddata")):
        os.environ["TESSDATA_PREFIX"] = str(LOCAL_TESSDATA)
        return "local"
    return ""


@lru_cache(maxsize=4)
def _langs(cmd: str, config: str) -> tuple[str, ...]:
    if config:
        return tuple(sorted(p.stem for p in LOCAL_TESSDATA.glob("*.traineddata")))
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return tuple(pytesseract.get_languages(config=""))
    except Exception:
        return ()


def status(s: dict) -> dict:
    if pytesseract is None:
        return {"ok": False, "error": "Falta la librería pytesseract (pip install -r requirements.txt)"}
    cmd = find_tesseract(s.get("tesseract_cmd", ""))
    if not cmd:
        return {"ok": False, "error": "Tesseract no está instalado: las páginas escaneadas y las palabras "
                                      "dentro de imágenes no se podrán buscar. Ver README."}
    langs = _langs(cmd, _config())  # también deja listo TESSDATA_PREFIX
    lang = _pick_lang(s.get("ocr_lang", "spa+eng"), langs)
    warn = None if "spa" in langs else "Tesseract no tiene el idioma español (spa): lee peor las tildes y la ñ."
    return {"ok": True, "cmd": cmd, "lang": lang, "langs": list(langs), "warning": warn}


def _pick_lang(wanted: str, have: tuple[str, ...]) -> str:
    ok = [l for l in wanted.split("+") if l in have]
    return "+".join(ok) or ("eng" if "eng" in have else (have[0] if have else "eng"))


def _clean_lines(text: str) -> list[str]:
    lines = [" ".join(l.split()).strip("|[]_ ") for l in text.splitlines()]
    # descarta líneas basura (1-2 caracteres sueltos que salen de dibujos)
    return [l for l in lines if sum(c.isalnum() for c in l) >= 3]


def _confident_lines(img, lang: str, psm: int, min_conf: int = 75) -> list[str]:
    """Solo palabras que Tesseract lee con buena confianza: evita la 'basura' que sale de los dibujos."""
    d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT,
                                  timeout=20)
    lines: dict[tuple, list[str]] = {}
    for i, w in enumerate(d["text"]):
        w = w.strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if w and conf >= min_conf and (sum(c.isalpha() for c in w) >= 2 or (w.isdigit() and len(w) >= 2)):
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(w)
    return [" ".join(ws) for ws in lines.values() if sum(len(x) for x in ws) >= 3]


def read(png: bytes, s: dict, bands: bool = False) -> str:
    """Texto de una imagen PNG. Devuelve '' si Tesseract no está disponible.
    bands=True (imágenes dentro del PDF): además se lee por franjas horizontales, porque en diagramas con
    cajas unidas por líneas Tesseract a veces confunde todo con un dibujo y se salta los textos de las cajas."""
    st = status(s)
    if not st["ok"]:
        return ""
    pytesseract.pytesseract.tesseract_cmd = st["cmd"]
    img = ImageOps.grayscale(Image.open(io.BytesIO(png)))
    if max(img.size) < 1000:  # las imágenes pequeñas se agrandan: Tesseract lee mejor
        f = 1000 / max(img.size)
        img = img.resize((int(img.width * f), int(img.height * f)))
    out: list[str] = []
    try:
        if bands:  # imagen del PDF: solo texto confiable (los dibujos generan letras falsas)
            out += _confident_lines(img, st["lang"], 3)
        else:      # página escaneada: es casi todo texto
            out += _clean_lines(pytesseract.image_to_string(img, lang=st["lang"], config="--psm 3", timeout=60))
        if bands and img.height <= 2400:
            if img.width > 1200:
                img = img.resize((1200, int(img.height * 1200 / img.width)))
            big = img.resize((img.width * 2, img.height * 2))
            step, tall = 140, 200            # franjas de 100 px (en tamaño original) que se solapan
            for y in range(0, big.height, step):
                band = big.crop((0, y, big.width, min(big.height, y + tall)))
                out += _confident_lines(band, st["lang"], 6)
    except Exception:
        pass
    # sin repetidos (las franjas se solapan)
    seen, uniq = set(), []
    for l in out:
        k = "".join(c for c in l.lower() if c.isalnum())
        if k and k not in seen and not any(k in x for x in seen):
            seen.add(k)
            uniq.append(l)
    return "\n".join(uniq).strip()
