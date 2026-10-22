"""API web de Multichat (FastAPI) + página en /static."""

import json
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ai, db, galaxy, indexer, ocr, rag
from . import pdf as pdfmod
from .config import BASE_DIR, DEFAULT_SETTINGS, IMG_DIR, KB_DIR, SECRET_KEYS, kind_of

import logging
from .config import DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(DATA_DIR / "multichat.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
for _n in ("multichat", "uvicorn.error"):
    logging.getLogger(_n).addHandler(_fh)
logging.getLogger("multichat").setLevel(logging.INFO)
log = logging.getLogger("multichat")

app = FastAPI(title="Multichat")
STATIC = BASE_DIR / "static"
SUBFOLDER = {"pdf": "pdfs", "audio": "audios", "video": "videos", "imagen": "imagenes"}


@app.on_event("startup")
def _startup() -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    for sub in SUBFOLDER.values():
        (KB_DIR / sub).mkdir(exist_ok=True)
    st = db.init()
    if st["ready"]:
        db.ensure_helpers()
        indexer.start()          # al arrancar indexa lo nuevo de la carpeta 'conocimiento'


@app.middleware("http")
async def no_cache(request: Request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/file"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _need_db() -> dict:
    st = db.state()
    if not st["ready"]:
        st = db.init()
        if st["ready"]:
            db.ensure_helpers()
    if not st["ready"]:
        raise HTTPException(503, st["error"] or "PostgreSQL no está listo")
    return db.get_settings()


# ------------------------------------------------------------------ estado y ajustes
@app.get("/api/status")
def status():
    st = db.state() if db.state()["ready"] else db.init()
    s = db.get_settings() if st["ready"] else dict(DEFAULT_SETTINGS)
    return {
        "db": {"ok": st["ready"], "pgvector": st["vector"], "error": st["error"],
               "stats": db.stats() if st["ready"] else None},
        "embed": ai.embed_status(s),
        "ollama_models": ai.ollama_models(s),
        "whisper": ai.whisper_ok(),
        "ocr": ocr.status(s),
        "chat_provider": s["chat_provider"],
        "kb_dir": str(KB_DIR),
        "index": _index_state(),
    }


@app.get("/api/settings")
def get_settings():
    s = _need_db()
    return {k: ("•" * 8 if k in SECRET_KEYS and v else v) for k, v in s.items()}


@app.post("/api/settings")
def save_settings(values: dict):
    _need_db()
    clean = {k: v for k, v in values.items() if k in DEFAULT_SETTINGS and not (k in SECRET_KEYS and set(str(v)) == {"•"})}
    db.save_settings(clean)
    return {"ok": True}


# ------------------------------------------------------------------ base de conocimiento
def _index_state() -> dict:
    s = dict(indexer.STATE)
    s["log"] = s["log"][-30:]
    return s


@app.get("/api/sources")
def sources():
    _need_db()
    return db.list_sources()


@app.post("/api/index")
def index(force: bool = False):
    _need_db()
    started = indexer.start(force)
    return {"started": started, "state": _index_state()}


@app.get("/api/index")
def index_state():
    return _index_state()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), kind: str | None = Form(None)):
    _need_db()
    hint, saved = kind, []
    for f in files:
        name = Path(f.filename or "archivo").name
        kind = kind_of(Path(name))
        ctype = (f.content_type or "").lower()
        if not kind and ctype.startswith(("audio/", "video/")):
            # extensión rara pero el navegador dice que es audio/video: se guarda con una extensión conocida
            kind = "audio" if ctype.startswith("audio/") else "video"
            name += ".m4a" if kind == "audio" else ".mp4"
        if not kind:
            raise HTTPException(400, f"'{name}' no es PDF, audio, video ni imagen.")
        if hint in SUBFOLDER and hint != kind:
            if {hint, kind} <= {"audio", "video"}:
                kind = hint                     # el usuario sabe si es un audio o un video
            else:
                nombres = {"pdf": "un PDF", "audio": "un audio", "video": "un video", "imagen": "una imagen"}
                raise HTTPException(400, f"Elegiste subir {nombres[hint]}, pero '{name}' es {nombres[kind]}.")
        dest = KB_DIR / SUBFOLDER[kind] / name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        real = kind_of(dest)                      # p. ej. un .mpeg que en realidad es solo audio
        if real and real != kind:
            dest = dest.replace(KB_DIR / SUBFOLDER[real] / name)
        saved.append(dest.relative_to(KB_DIR).as_posix())
    indexer.start()
    return {"saved": saved}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: int):
    """Quita un archivo de la base de conocimiento: lo borra de la carpeta y de PostgreSQL."""
    _need_db()
    if indexer.STATE["running"]:
        raise HTTPException(409, "Espera a que termine la indexación y vuelve a intentarlo.")
    src = db.get_source(source_id)
    if not src:
        raise HTTPException(404, "Ese archivo ya no está en la base de conocimiento.")
    path = (KB_DIR / src["path"]).resolve()
    if KB_DIR.resolve() in path.parents and path.is_file():
        path.unlink()
    for img in db.delete_source(source_id):
        (IMG_DIR / img).unlink(missing_ok=True)
    log.info("Archivo eliminado: %s", src["path"])
    return {"ok": True, "name": src["name"]}


@app.get("/api/file/{source_id}")
def file(source_id: int):
    _need_db()
    src = db.get_source(source_id)
    path = (KB_DIR / src["path"]).resolve() if src else None
    if not path or not path.is_file() or KB_DIR.resolve() not in path.parents:
        raise HTTPException(404, "Archivo no encontrado")
    return FileResponse(path)


@app.get("/api/page/{source_id}/{page}.png")
def page_png(source_id: int, page: int, q: str | None = None, w: str | None = None, dpi: int = 90):
    _need_db()
    src = db.get_source(source_id)
    if not src or src["kind"] != "pdf":
        raise HTTPException(404)
    words = [x for x in (w or "").split("|") if x.strip()][:8]      # palabras buscadas, separadas por |
    dpi = max(60, min(dpi, 220))      # el lector con zoom pide más resolución para que la letra se lea nítida
    png = pdfmod.render_page(KB_DIR / src["path"], page, highlight=q, dpi=dpi, words=words)
    return Response(png, media_type="image/png")


@app.get("/api/image/{chunk_id}.png")
def chunk_image(chunk_id: int):
    _need_db()
    ch = db.chunk(chunk_id)
    if not ch or not ch["image_path"]:
        raise HTTPException(404)
    return FileResponse(IMG_DIR / ch["image_path"])


# ------------------------------------------------------------------ consulta
class Ask(BaseModel):
    question: str
    kinds: list[str] | None = None
    source_id: int | None = None      # preguntar a un solo archivo
    conv: str | None = None           # conversación general ("＋ Nuevo chat")


@app.post("/api/search")
def search(body: Ask):
    """Solo búsqueda semántica (sin redactar respuesta): muestra los fragmentos más parecidos."""
    s = _need_db()
    try:
        hits = rag.retrieve(body.question, s, body.kinds, k=8, source_id=body.source_id)
        rag.locate(body.question, hits, s)
    except ai.AIError as e:
        raise HTTPException(503, str(e))
    return [rag.public(h) for h in hits]


@app.post("/api/ask")
def ask(body: Ask):
    s = _need_db()
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Escribe una pregunta")

    # con @archivo dentro de una conversación: se busca en ese archivo pero se guarda en la conversación
    keep = None if body.conv else body.source_id

    def stream():
        t0 = time.time()
        trace: dict = {}
        try:
            if body.source_id:
                hits = rag.retrieve_file(question, s, body.source_id, trace=trace)
            else:
                hits = rag.retrieve(question, s, body.kinds, trace=trace)
            rag.locate(question, hits, s, trace.get("qvec"))      # minuto y segundo exactos
        except ai.AIError as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
            return
        try:
            yield json.dumps(galaxy.trace_event(trace, hits, s)) + "\n"
        except Exception:  # noqa: BLE001  (el modo cerebro es extra: si falla, la respuesta sigue)
            pass
        srcs = [rag.public(h) for h in hits]
        yield json.dumps({"type": "sources", "sources": srcs, "search_s": round(time.time() - t0, 1)}) + "\n"
        if not hits:
            msg = "La base de conocimiento está vacía o todavía se está indexando."
            yield json.dumps({"type": "token", "text": msg}) + "\n"
            db.add_message("user", question, source_id=keep, conv=body.conv)
            db.add_message("assistant", msg, [], keep, body.conv)
            yield json.dumps({"type": "done"}) + "\n"
            return
        history = db.messages(8, keep, body.conv)
        answer = []
        try:
            target = db.get_source(body.source_id) if body.source_id else None
            for piece in ai.chat_stream(rag.build_messages(question, hits, history, target), s):
                answer.append(piece)
                yield json.dumps({"type": "token", "text": piece}) + "\n"
        except ai.AIError as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        db.add_message("user", question, source_id=keep, conv=body.conv)
        db.add_message("assistant", "".join(answer).strip(), srcs, keep, body.conv)
        yield json.dumps({"type": "done", "total_s": round(time.time() - t0, 1)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/map")
def galaxy_map():
    """Puntos de la galaxia del conocimiento (embeddings proyectados a 2D)."""
    s = _need_db()
    return {"points": galaxy.points(s)}


@app.post("/api/trace")
def galaxy_trace(body: Ask):
    """Ubica una consulta en la galaxia y devuelve sus vecinos (sin redactar respuesta)."""
    s = _need_db()
    trace: dict = {}
    try:
        hits = rag.retrieve(body.question, s, body.kinds, k=8, trace=trace, source_id=body.source_id)
        rag.locate(body.question, hits, s, trace.get("qvec"))
    except ai.AIError as e:
        raise HTTPException(503, str(e))
    return {"trace": galaxy.trace_event(trace, hits, s), "hits": [rag.public(h) for h in hits]}


@app.post("/api/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    """Pregunta por voz: el navegador graba, Whisper (o Gemini) lo pasa a texto."""
    s = _need_db()
    import tempfile
    suffix = Path(audio.filename or "voz.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        segs, _ = ai.transcribe(path, s)
    except Exception as e:  # noqa: BLE001
        log.exception("Error transcribiendo la voz")
        raise HTTPException(503, f"No se pudo transcribir ({type(e).__name__}): {e}")
    finally:
        Path(path).unlink(missing_ok=True)
    return {"text": " ".join(x["text"] for x in segs).strip()}


@app.get("/api/messages")
def messages(source_id: int | None = None, conv: str | None = None):
    _need_db()
    return db.messages(source_id=source_id, conv=conv)


@app.delete("/api/messages")
def clear(source_id: int | None = None, conv: str | None = None):
    _need_db()
    db.clear_messages(source_id, conv)
    return {"ok": True}


@app.get("/api/conversations")
def conversations():
    _need_db()
    return db.conversations()


@app.exception_handler(Exception)
async def errors(_: Request, exc: Exception):
    return JSONResponse({"detail": str(exc)[:400]}, status_code=500)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
