"""Indexador de la base de conocimiento: recorre la carpeta 'conocimiento', procesa cada archivo nuevo o
modificado y guarda sus fragmentos con embedding en PostgreSQL.

  PDF     → texto por página + OCR de páginas escaneadas + cada imagen descrita por IA y leída con OCR
  Imagen  → descripción por IA + OCR
  Audio   → transcripción con marcas de tiempo (faster-whisper) en trozos de ~30 s
  Video   → transcripción de su audio + fotogramas descritos por IA (con su minuto)
"""

import logging
import hashlib
import io
import threading
import time
import traceback
from pathlib import Path

from . import ai, db, ocr
from . import pdf as pdfmod
from .config import (AUDIO_WINDOW_S, CHUNK_OVERLAP, CHUNK_SIZE, IMG_DIR, KB_DIR, MAX_IMAGES_PER_PDF, MAX_PAGES,
                     MIN_IMAGE_PX, kind_of)

STATE = {"running": False, "file": "", "step": "", "done": 0, "total": 0, "log": [], "started": None,
         "finished": None}
_lock = threading.Lock()


def _log(msg: str) -> None:
    logging.getLogger("multichat").info("[indexar] %s", msg)
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    STATE["log"] = STATE["log"][-200:]


def _step(msg: str) -> None:
    STATE["step"] = msg


def fmt_time(t: float | None) -> str:
    if t is None:
        return ""
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan() -> list[Path]:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in KB_DIR.rglob("*") if p.is_file() and kind_of(p) and not p.name.startswith("~$"))


_again = {"on": False}


def start(force: bool = False) -> bool:
    """Lanza la indexación en segundo plano. Devuelve False si ya estaba corriendo."""
    with _lock:
        if STATE["running"]:
            _again["on"] = True      # llegó un archivo nuevo mientras indexaba: repetir al terminar
            return False
        STATE.update(running=True, file="", step="Revisando la carpeta…", done=0, total=0, log=[],
                     started=time.time(), finished=None)
    threading.Thread(target=_run, args=(force,), daemon=True).start()
    return True


def _run(force: bool) -> None:
    try:
        s = db.get_settings()
        files = scan()
        rel = {p: p.relative_to(KB_DIR).as_posix() for p in files}
        removed = db.delete_missing(set(rel.values()))
        if removed:
            _log(f"Se quitaron {removed} archivo(s) que ya no están en la carpeta.")
        known = db.sources_by_path()
        stale = db.without_segments()
        todo, retime = [], []
        for p in files:
            old = known.get(rel[p])
            digest = sha1_of(p)
            if force or not old or old["sha1"] != digest or old["status"] != "listo" \
                    or old["embed_model"] != ai.embed_label(s):
                todo.append((p, digest))
            elif old["id"] in stale:
                retime.append((p, old["id"]))
        STATE["total"] = len(todo) + len(retime)
        for p, sid in retime:        # indexados antes de guardar el segundo exacto de cada frase
            STATE["file"] = rel[p]
            _log(f"Guardando el segundo exacto de cada frase: {rel[p]}")
            try:
                _step("Transcribiendo de nuevo (solo el audio)…")
                segs, _ = ai.transcribe(str(p), s)
                if not segs:
                    raise ValueError("Whisper no encontró voz esta vez")
                rows = _windows(segs, "Audio, minuto" if kind_of(p) == "audio" else "Narración del video, minuto")
                _step(f"Calculando embeddings de {len(rows)} fragmentos (EmbeddingGemma)…")
                db.replace_transcription(sid, rows, ai.embed([ai.doc_prompt(r["content"], p.stem) for r in rows], s))
                _log(f"✔ {rel[p]}: {sum(len(r['segments']) for r in rows)} frases con su minuto y segundo")
            except Exception as e:  # noqa: BLE001  (si falla, se queda la transcripción anterior)
                _log(f"✖ {rel[p]}: {e}")
                logging.getLogger("multichat").exception("Error guardando los tiempos")
            STATE["done"] += 1
        if not todo and not retime:
            _log(f"Todo al día: {len(files)} archivo(s) ya estaban indexados.")
        for p, digest in todo:
            STATE["file"] = rel[p]
            _log(f"Procesando {rel[p]}")
            sid = db.upsert_source(rel[p], p.name, kind_of(p), digest, p.stat().st_size)
            try:
                rows, extra = _process(p, kind_of(p), sid, s)
                if not rows:
                    raise ValueError("no se encontró contenido (¿archivo vacío o sin audio?)")
                _step(f"Calculando embeddings de {len(rows)} fragmentos (EmbeddingGemma)…")
                vecs = ai.embed([ai.doc_prompt(r["content"], p.stem) for r in rows], s)
                db.add_chunks(sid, rows, vecs)
                db.finish_source(sid, "listo", embed_model=ai.embed_label(s), **extra)
                _log(f"✔ {rel[p]}: {len(rows)} fragmentos")
            except Exception as e:  # noqa: BLE001
                db.finish_source(sid, "error", error=str(e)[:400])
                _log(f"✖ {rel[p]}: {e}")
                logging.getLogger("multichat").exception("Error indexando")
            STATE["done"] += 1
    except Exception as e:  # noqa: BLE001
        _log(f"✖ Error general: {e}")
        logging.getLogger("multichat").exception("Error indexando")
    finally:
        STATE.update(running=False, file="", step="", finished=time.time())
        if _again["on"]:
            _again["on"] = False
            start()


# ---------------------------------------------------------------------------- por tipo
def _process(path: Path, kind: str, sid: int, s: dict) -> tuple[list[dict], dict]:
    if kind == "pdf":
        return _pdf(path, sid, s)
    if kind == "imagen":
        png = _to_png(path.read_bytes())
        return [_visual_chunk(png, s, sid, "img", kind="imagen", label="Imagen")], {}
    if kind == "audio":
        return _audio(path, s)
    if kind == "video":
        return _video(path, sid, s)
    return [], {}


def _pdf(path: Path, sid: int, s: dict) -> tuple[list[dict], dict]:
    _step("Leyendo el PDF…")
    parsed = pdfmod.parse(path.read_bytes(), MAX_PAGES, MAX_IMAGES_PER_PDF, MIN_IMAGE_PX)
    pages = [(n, t) for n, t in parsed.pages if t]
    for i, sc in enumerate(parsed.scanned, 1):
        _step(f"OCR de página escaneada {i}/{len(parsed.scanned)}…")
        txt = ocr.read(sc.png, s)
        if txt:
            pages.append((sc.page, txt))
    rows = [{"kind": "texto", "page": n, "content": t} for n, t in pdfmod.chunk_pages(sorted(pages), CHUNK_SIZE, CHUNK_OVERLAP)]
    for i, im in enumerate(parsed.images, 1):
        _step(f"Mirando imagen {i}/{len(parsed.images)} (página {im.page})…")
        rows.append(_visual_chunk(im.png, s, sid, f"p{im.page}_{i}", kind="imagen", page=im.page,
                                  label=f"Imagen de la página {im.page}"))
    return rows, {"pages": parsed.total_pages}


def _visual_chunk(png: bytes, s: dict, sid: int, tag: str, kind: str, label: str, page: int | None = None,
                  t: float | None = None) -> dict:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMG_DIR / f"{sid}_{tag}.png"
    img_path.write_bytes(png)
    desc = ""
    try:
        desc = ai.describe_image(png, s)
    except ai.AIError as e:
        _log(f"   (no se pudo describir la imagen: {e})")
    text = ocr.read(png, s, bands=True)
    content = f"{label}. Descripción: {desc or 'sin descripción'}."
    if text:
        content += f" Texto visible en la imagen: {text}"
    return {"kind": kind, "page": page, "t_start": t, "t_end": t, "content": content,
            "image_path": img_path.name}


def _windows(segs: list[dict], prefix: str) -> list[dict]:
    """Agrupa los segmentos de la transcripción en trozos de ~30 s (cada uno con su minuto de inicio)."""
    rows, cur = [], []
    for sg in segs:
        cur.append(sg)
        if cur[-1]["end"] - cur[0]["start"] >= AUDIO_WINDOW_S:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    return [{"kind": "transcripcion", "t_start": w[0]["start"], "t_end": w[-1]["end"],
             "content": f"{prefix} [{fmt_time(w[0]['start'])}–{fmt_time(w[-1]['end'])}]: "
                        + " ".join(x["text"] for x in w),
             # cada frase con su segundo exacto: así se puede decir en qué minuto y segundo se dice algo
             "segments": [{"start": round(x["start"], 1), "end": round(x["end"], 1), "text": x["text"]} for x in w]}
            for w in rows]


def _audio(path: Path, s: dict) -> tuple[list[dict], dict]:
    _step("Transcribiendo el audio (faster-whisper)… puede tardar")
    segs, dur = ai.transcribe(str(path), s)
    if not segs:
        raise ValueError("Whisper no encontró voz en el audio (¿está en silencio o muy bajo?)")
    return _windows(segs, "Audio, minuto"), {"duration": dur}


def _video(path: Path, sid: int, s: dict) -> tuple[list[dict], dict]:
    rows, dur, why = [], None, ""
    _step("Transcribiendo el audio del video…")
    try:
        segs, dur = ai.transcribe(str(path), s)
        rows += _windows(segs, "Narración del video, minuto")
    except Exception as e:  # noqa: BLE001
        why = str(e)
        _log(f"   (sin audio o no se pudo transcribir: {e})")
    else:
        why = "el audio no tiene voz que se pueda transcribir"
    frames, dur2 = extract_frames(path, int(s.get("video_frames", 12) or 12))
    dur = dur or dur2
    if not rows and not frames:
        raise ValueError(why)
    for i, (t, png) in enumerate(frames, 1):
        _step(f"Mirando fotograma {i}/{len(frames)} ({fmt_time(t)})…")
        rows.append(_visual_chunk(png, s, sid, f"f{int(t)}", kind="escena", t=t,
                                  label=f"Escena del video en el minuto {fmt_time(t)}"))
    return rows, {"duration": dur}


def extract_frames(path: Path, n: int) -> tuple[list[tuple[float, bytes]], float | None]:
    """Las escenas DISTINTAS del video (hasta n), no fotogramas al azar: así no se escapan las imágenes que se
    insertan unos segundos. Se mira el video cada medio segundo; la escena más común (la persona que habla o la
    diapositiva principal) es el 'fondo', y cada tramo que se aleja de él (una imagen, otra diapositiva) es una
    escena nueva. De cada tramo se toma el fotograma del medio y se recortan las franjas negras."""
    try:
        import av
        import numpy as np
    except ImportError:
        return [], None
    with av.open(str(path)) as cont:
        stream = next((st for st in cont.streams if st.type == "video"), None)
        if stream is None:
            return [], None
        dur = float(cont.duration / 1_000_000) if cont.duration else None
        if not dur and stream.duration and stream.time_base:
            dur = float(stream.duration * stream.time_base)
        # 1) cada medio segundo, una miniatura; la zona con imagen (sin franjas negras) es la unión de todas
        ts, thumbs, box, nxt = [], [], None, 0.0
        for frame in cont.decode(stream):
            if frame.time is None or frame.time < nxt:
                continue
            nxt = frame.time + 0.5
            im = frame.to_image().convert("RGB")
            im.thumbnail((160, 160))
            bb = im.convert("L").point(lambda v: 255 if v > 18 else 0).getbbox()
            if bb:
                f = [bb[0] / im.width, bb[1] / im.height, bb[2] / im.width, bb[3] / im.height]
                box = f if box is None else [min(box[0], f[0]), min(box[1], f[1]), max(box[2], f[2]), max(box[3], f[3])]
            ts.append(float(frame.time)); thumbs.append(im)
        dur = dur or (ts[-1] if ts else 0)
        if not ts:
            return [], dur or None
        if not box or (box[2] - box[0]) * (box[3] - box[1]) > 0.9:
            box = None
        px, hist = [], []          # huella de la zona con imagen: brillo (32x32) y colores
        for im in thumbs:
            if box:
                im = im.crop((int(box[0] * im.width), int(box[1] * im.height), int(box[2] * im.width), int(box[3] * im.height)))
            hsv = np.asarray(im.resize((64, 64)).convert("HSV"), dtype=np.int32)
            h = np.histogram2d(hsv[..., 0].ravel(), hsv[..., 2].ravel(), bins=(12, 8), range=((0, 256), (0, 256)))[0]
            px.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)); hist.append(h.ravel() / h.sum())
        del thumbs
        # 2) distancia de cada momento al 'fondo' y tramos que se alejan de él
        dist = np.abs(px - np.median(px, axis=0)).mean((1, 2))
        far = (dist > 15) | (np.abs(hist - np.median(hist, axis=0)).sum(1) > 0.3)
        picks = [(int(np.argmin(dist)), 0.0)]                       # el fondo, una sola vez
        i = 0
        while i < len(ts):
            if far[i]:
                j = i
                while j + 1 < len(ts) and far[j + 1]:
                    j += 1
                picks.append(((i + j) // 2, float(dist[i:j + 1].mean()) * (j - i + 1)))
                i = j + 1
            else:
                i += 1
        # 3) sin escenas repetidas y como máximo n (se quedan las más distintas y largas)
        uniq = []
        for k, score in picks:
            if all(np.abs(px[k] - px[u]).mean() > 12 or np.abs(hist[k] - hist[u]).sum() > 0.3 for u, _ in uniq):
                uniq.append((k, score))
        uniq = [uniq[0]] + sorted(uniq[1:], key=lambda p: -p[1])[: max(n - 1, 0)]
        times = sorted(ts[k] for k, _ in uniq)
        # 4) el fotograma de cada escena en tamaño real, sin las franjas negras
        out = []
        for t in times:
            try:
                cont.seek(int(t / stream.time_base), stream=stream, any_frame=False, backward=True)
                for frame in cont.decode(stream):
                    if frame.time is None or frame.time >= t - 0.05:
                        img = frame.to_image()
                        if box:
                            img = img.crop((int(box[0] * img.width), int(box[1] * img.height),
                                            int(box[2] * img.width), int(box[3] * img.height)))
                        img.thumbnail((1280, 1280))
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        out.append((round(float(frame.time if frame.time is not None else t), 1), buf.getvalue()))
                        break
            except Exception:  # noqa: BLE001
                continue
    return out, dur or None


def _to_png(data: bytes) -> bytes:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()
