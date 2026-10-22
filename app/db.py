"""PostgreSQL: fuentes (archivos de la base de conocimiento), fragmentos con su embedding, ajustes y chat.

Si el servidor tiene la extensión pgvector, el embedding se guarda como `vector(768)` y la búsqueda por
similitud la hace PostgreSQL (operador <=>). Si no la tiene, se guarda como `real[]` y la similitud
coseno se calcula en Python con numpy. En ambos casos TODO queda guardado en PostgreSQL."""

import json
import threading
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL, DEFAULT_SETTINGS, EMBED_DIM

_lock = threading.Lock()
_state = {"ready": False, "vector": False, "error": None}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id          SERIAL PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,          -- ruta relativa dentro de la carpeta 'conocimiento'
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,                 -- pdf | audio | video | imagen
    sha1        TEXT,
    size_bytes  BIGINT,
    pages       INT,
    duration_s  REAL,
    status      TEXT NOT NULL DEFAULT 'pendiente',   -- pendiente | indexando | listo | error
    error       TEXT,
    embed_model TEXT,
    indexed_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    source_id   INT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                 -- texto | imagen | transcripcion | escena
    page        INT,                           -- página (PDF)
    t_start     REAL,                          -- segundo inicial (audio / video)
    t_end       REAL,
    content     TEXT NOT NULL,
    image_path  TEXT,                          -- imagen o fotograma asociado
    embedding   {vec_type}
);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source_id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id         SERIAL PRIMARY KEY,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    sources    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- conversación con un solo archivo (NULL = chat general); se borra junto con el archivo
ALTER TABLE messages ADD COLUMN IF NOT EXISTS source_id INT REFERENCES sources(id) ON DELETE CASCADE;
-- conversaciones generales ("＋ Nuevo chat"); las anteriores quedan en una conversación llamada 'general'
ALTER TABLE messages ADD COLUMN IF NOT EXISTS conv TEXT;
UPDATE messages SET conv = 'general' WHERE conv IS NULL AND source_id IS NULL;
-- frases de la transcripción con su segundo exacto: [{"start", "end", "text"}]
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS segments JSONB;
"""


def _with_db(url: str, dbname: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/" + dbname, p.query, p.fragment))


def _ensure_database() -> None:
    """Crea la base de datos 'multichat' si todavía no existe (conectándose a la base 'postgres')."""
    target = urlsplit(DATABASE_URL).path.lstrip("/") or "multichat"
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=5).close()
        return
    except psycopg.OperationalError as e:
        if "does not exist" not in str(e) and "no existe" not in str(e):
            raise
    with psycopg.connect(_with_db(DATABASE_URL, "postgres"), autocommit=True, connect_timeout=5) as c:
        c.execute(f'CREATE DATABASE "{target}"')


@contextmanager
def connect():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> dict:
    """Crea la base de datos y las tablas. Devuelve el estado (ready, vector, error)."""
    with _lock:
        try:
            _ensure_database()
            with connect() as c:
                vector = False
                try:
                    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    vector = True
                except psycopg.Error:
                    c.rollback()
                # si la tabla ya existía, se respeta el tipo con que se creó
                row = c.execute("SELECT data_type, udt_name FROM information_schema.columns "
                                "WHERE table_name='chunks' AND column_name='embedding'").fetchone()
                if row:
                    vector = row["udt_name"] == "vector"
                vec_type = f"vector({EMBED_DIM})" if vector else "real[]"
                c.execute(SCHEMA.replace("{vec_type}", vec_type))
            _state.update(ready=True, vector=vector, error=None)
        except Exception as e:  # noqa: BLE001
            _state.update(ready=False, error=_friendly(e))
    return dict(_state)


def _friendly(e: Exception) -> str:
    msg = str(e)
    if "password authentication failed" in msg or "autentificación" in msg or "autenticación" in msg:
        return "PostgreSQL rechazó la contraseña. Corrige DATABASE_URL en el archivo .env."
    if "Connection refused" in msg or "could not connect" in msg or "No se puede establecer" in msg \
            or "timeout" in msg.lower():
        return ("No me pude conectar a PostgreSQL en localhost:5432. Abre 'Servicios' de Windows e inicia "
                "el servicio postgresql-x64-17.")
    return msg[:300]


def state() -> dict:
    return dict(_state)


# ---------------------------------------------------------------- ajustes
def get_settings(include_secrets: bool = True) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if _state["ready"]:
        with connect() as c:
            for r in c.execute("SELECT key, value FROM settings"):
                if r["key"] in s:
                    s[r["key"]] = r["value"]
    return s


def save_settings(values: dict) -> None:
    with connect() as c:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS and v is not None:
                c.execute("INSERT INTO settings(key, value) VALUES (%s, %s) "
                          "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, str(v)))


# ---------------------------------------------------------------- fuentes
def list_sources() -> list[dict]:
    with connect() as c:
        return c.execute("""
            SELECT s.id, s.path, s.name, s.kind, s.size_bytes, s.pages, s.duration_s, s.status, s.error,
                   s.embed_model, s.indexed_at,
                   COUNT(ch.id) AS chunks,
                   COUNT(ch.id) FILTER (WHERE ch.kind IN ('imagen','escena')) AS visuals
            FROM sources s LEFT JOIN chunks ch ON ch.source_id = s.id
            GROUP BY s.id ORDER BY s.kind, s.name""").fetchall()


def get_source(source_id: int) -> dict | None:
    with connect() as c:
        return c.execute("SELECT * FROM sources WHERE id = %s", (source_id,)).fetchone()


def sources_by_path() -> dict[str, dict]:
    with connect() as c:
        return {r["path"]: r for r in c.execute("SELECT * FROM sources")}


def upsert_source(path: str, name: str, kind: str, sha1: str, size: int) -> int:
    with connect() as c:
        r = c.execute("""INSERT INTO sources(path, name, kind, sha1, size_bytes, status)
                         VALUES (%s, %s, %s, %s, %s, 'indexando')
                         ON CONFLICT (path) DO UPDATE SET name=EXCLUDED.name, kind=EXCLUDED.kind, sha1=EXCLUDED.sha1,
                             size_bytes=EXCLUDED.size_bytes, status='indexando', error=NULL
                         RETURNING id""", (path, name, kind, sha1, size)).fetchone()
        c.execute("DELETE FROM chunks WHERE source_id = %s", (r["id"],))
        return r["id"]


def finish_source(source_id: int, status: str, error: str | None = None, pages: int | None = None,
                  duration: float | None = None, embed_model: str | None = None) -> None:
    with connect() as c:
        c.execute("""UPDATE sources SET status=%s, error=%s, pages=COALESCE(%s, pages),
                     duration_s=COALESCE(%s, duration_s), embed_model=COALESCE(%s, embed_model),
                     indexed_at=CASE WHEN %s='listo' THEN now() ELSE indexed_at END WHERE id=%s""",
                  (status, error, pages, duration, embed_model, status, source_id))


def delete_source(source_id: int) -> list[str]:
    """Borra un archivo de la base (sus fragmentos se borran en cascada). Devuelve las imágenes que tenía."""
    with connect() as c:
        imgs = [r["image_path"] for r in c.execute(
            "SELECT image_path FROM chunks WHERE source_id = %s AND image_path IS NOT NULL", (source_id,)).fetchall()]
        c.execute("DELETE FROM sources WHERE id = %s", (source_id,))
    _cache.clear()
    return imgs


def delete_missing(paths_on_disk: set[str]) -> int:
    with connect() as c:
        rows = c.execute("SELECT id, path FROM sources").fetchall()
        gone = [r["id"] for r in rows if r["path"] not in paths_on_disk]
        if gone:
            c.execute("DELETE FROM sources WHERE id = ANY(%s)", (gone,))
            _cache.clear()
        return len(gone)


# ---------------------------------------------------------------- fragmentos
def _vec_param(v: list[float]):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]" if _state["vector"] else [float(x) for x in v]


def add_chunks(source_id: int, rows: list[dict], vectors: list[list[float]]) -> None:
    cast = "::vector" if _state["vector"] else ""
    with connect() as c:
        with c.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO chunks(source_id, kind, page, t_start, t_end, content, image_path, segments, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s{cast})""",
                [(source_id, r["kind"], r.get("page"), r.get("t_start"), r.get("t_end"), r["content"],
                  r.get("image_path"), json.dumps(r["segments"]) if r.get("segments") else None, _vec_param(v))
                 for r, v in zip(rows, vectors)])
    _cache.clear()


def without_segments() -> set[int]:
    """Archivos indexados antes de guardar el segundo exacto de cada frase."""
    with connect() as c:
        return {r["source_id"] for r in c.execute(
            "SELECT DISTINCT source_id FROM chunks WHERE kind = 'transcripcion' AND segments IS NULL").fetchall()}


def replace_transcription(source_id: int, rows: list[dict], vectors: list[list[float]]) -> None:
    """Cambia solo la transcripción de un archivo (las escenas del video ya descritas se conservan)."""
    with connect() as c:
        c.execute("DELETE FROM chunks WHERE source_id = %s AND kind = 'transcripcion'", (source_id,))
    add_chunks(source_id, rows, vectors)


_cache: dict = {}


def _matrix(embed_model: str):
    """Sin pgvector: carga todos los embeddings una vez (se invalida al indexar)."""
    if embed_model in _cache:
        return _cache[embed_model]
    with connect() as c:
        rows = c.execute("""SELECT ch.id, ch.embedding FROM chunks ch JOIN sources s ON s.id = ch.source_id
                            WHERE s.embed_model = %s AND s.status = 'listo'""", (embed_model,)).fetchall()
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    mat = np.array([r["embedding"] for r in rows], dtype=np.float32) if rows else np.zeros((0, EMBED_DIM), np.float32)
    if len(mat):
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    _cache[embed_model] = (ids, mat)
    return ids, mat


def nearest(query_vec: list[float], embed_model: str, kinds: list[str] | None, limit: int,
            source_id: int | None = None) -> list[dict]:
    """Los `limit` fragmentos más parecidos (similitud coseno) a la pregunta."""
    kinds = kinds or ["pdf", "audio", "video", "imagen"]
    if _state["vector"]:
        with connect() as c:
            return c.execute("""
                SELECT ch.id, ch.kind, ch.page, ch.t_start, ch.t_end, ch.content, ch.image_path, ch.segments, ch.source_id,
                       s.name, s.kind AS source_kind, s.path, 1 - (ch.embedding <=> %s::vector) AS score
                FROM chunks ch JOIN sources s ON s.id = ch.source_id
                WHERE s.status = 'listo' AND s.embed_model = %s AND (s.kind = ANY(%s) OR (ch.kind = 'imagen' AND 'imagen' = ANY(%s))) AND (%s::int IS NULL OR s.id = %s::int)
                ORDER BY ch.embedding <=> %s::vector LIMIT %s""",
                (_vec_param(query_vec), embed_model, kinds, kinds, source_id, source_id, _vec_param(query_vec), limit)).fetchall()
    ids, mat = _matrix(embed_model)
    if not len(ids):
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    scores = mat @ q
    with connect() as c:   # primero se filtra por tipo (así 'Audios' o 'Imágenes' nunca quedan vacíos por culpa de los PDF)
        allowed = {r["id"] for r in c.execute("""
            SELECT ch.id FROM chunks ch JOIN sources s ON s.id = ch.source_id WHERE """ + "(s.kind = ANY(%s) OR (ch.kind = 'imagen' AND 'imagen' = ANY(%s))) AND (%s::int IS NULL OR s.id = %s::int)", (kinds, kinds, source_id, source_id)).fetchall()}
    order = [i for i in np.argsort(-scores) if int(ids[i]) in allowed][: limit * 2]
    with connect() as c:
        rows = c.execute("""
            SELECT ch.id, ch.kind, ch.page, ch.t_start, ch.t_end, ch.content, ch.image_path, ch.segments, ch.source_id,
                   s.name, s.kind AS source_kind, s.path
            FROM chunks ch JOIN sources s ON s.id = ch.source_id
            WHERE ch.id = ANY(%s)""", ([int(ids[i]) for i in order],)).fetchall()
    by_id = {r["id"]: r for r in rows}
    out = []
    for i in order:
        r = by_id.get(int(ids[i]))
        if r:
            r["score"] = float(scores[i])
            out.append(r)
        if len(out) >= limit:
            break
    return out


def source_chunks(source_id: int) -> list[dict]:
    """Todos los fragmentos de un archivo, en orden (página / minuto)."""
    with connect() as c:
        return c.execute("""
            SELECT ch.id, ch.kind, ch.page, ch.t_start, ch.t_end, ch.content, ch.image_path, ch.segments, ch.source_id,
                   s.name, s.kind AS source_kind, s.path
            FROM chunks ch JOIN sources s ON s.id = ch.source_id
            WHERE s.id = %s ORDER BY ch.page NULLS FIRST, ch.t_start NULLS FIRST, ch.id""", (source_id,)).fetchall()


def keyword_hits(words: list[str], kinds: list[str] | None, limit: int = 30, source_id: int | None = None) -> list[dict]:
    """Fragmentos que contienen literalmente alguna palabra de la pregunta (refuerza la búsqueda semántica)."""
    if not words:
        return []
    kinds = kinds or ["pdf", "audio", "video", "imagen"]
    pattern = "|".join(words)
    with connect() as c:
        return c.execute("""
            SELECT ch.id, ch.kind, ch.page, ch.t_start, ch.t_end, ch.content, ch.image_path, ch.segments, ch.source_id,
                   s.name, s.kind AS source_kind, s.path
            FROM chunks ch JOIN sources s ON s.id = ch.source_id
            WHERE s.status = 'listo' AND (s.kind = ANY(%s) OR (ch.kind = 'imagen' AND 'imagen' = ANY(%s))) AND (%s::int IS NULL OR s.id = %s::int) AND unaccent_lower(ch.content) ~ %s
            LIMIT %s""", (kinds, kinds, source_id, source_id, pattern, limit)).fetchall()


def ensure_helpers() -> None:
    """Función SQL para buscar sin importar mayúsculas ni tildes (no necesita extensiones)."""
    with connect() as c:
        c.execute(r"""CREATE OR REPLACE FUNCTION unaccent_lower(t text) RETURNS text
                      LANGUAGE sql IMMUTABLE AS $$
                      SELECT translate(lower(t), 'áéíóúüñàèìòù', 'aeiouunaeiou') $$""")


def chunk(chunk_id: int) -> dict | None:
    with connect() as c:
        return c.execute("SELECT * FROM chunks WHERE id = %s", (chunk_id,)).fetchone()


def map_rows(embed_model: str) -> list[dict]:
    """Todos los fragmentos con su embedding, para dibujar la 'galaxia del conocimiento'."""
    emb = "ch.embedding::real[]" if _state["vector"] else "ch.embedding"
    with connect() as c:
        return c.execute(f"""
            SELECT ch.id, ch.kind, ch.page, ch.t_start, ch.source_id, LEFT(ch.content, 180) AS content,
                   s.name, s.kind AS source_kind, {emb} AS emb
            FROM chunks ch JOIN sources s ON s.id = ch.source_id
            WHERE s.status = 'listo' AND s.embed_model = %s ORDER BY ch.id""", (embed_model,)).fetchall()


def stats() -> dict:
    with connect() as c:
        r = c.execute("""SELECT COUNT(*) FILTER (WHERE kind='pdf') AS pdf, COUNT(*) FILTER (WHERE kind='audio') AS audio,
                                COUNT(*) FILTER (WHERE kind='video') AS video, COUNT(*) FILTER (WHERE kind='imagen') AS imagen,
                                COUNT(*) FILTER (WHERE status='listo') AS listos FROM sources""").fetchone()
        r["chunks"] = c.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return r


# ---------------------------------------------------------------- chat
def add_message(role: str, content: str, sources: list | None = None, source_id: int | None = None,
                conv: str | None = None) -> None:
    with connect() as c:
        c.execute("INSERT INTO messages(role, content, sources, source_id, conv) VALUES (%s, %s, %s, %s, %s)",
                  (role, content, json.dumps(sources) if sources is not None else None, source_id,
                   None if source_id else (conv or "general")))


def _scope(source_id: int | None, conv: str | None) -> tuple[str, tuple]:
    """Una conversación con un archivo (source_id) o una conversación general (conv)."""
    if source_id:
        return "source_id = %s", (source_id,)
    return "source_id IS NULL AND conv = %s", (conv or "general",)


def messages(limit: int = 60, source_id: int | None = None, conv: str | None = None) -> list[dict]:
    """Historial de una conversación general o de la conversación con un archivo."""
    where, args = _scope(source_id, conv)
    with connect() as c:
        rows = c.execute(f"SELECT role, content, sources FROM messages WHERE {where} ORDER BY id DESC LIMIT %s",
                         (*args, limit)).fetchall()
    return rows[::-1]


def clear_messages(source_id: int | None = None, conv: str | None = None) -> None:
    where, args = _scope(source_id, conv)
    with connect() as c:
        c.execute(f"DELETE FROM messages WHERE {where}", args)


def conversations() -> list[dict]:
    """Las conversaciones para la barra lateral, la más reciente primero: las generales (título = su primera
    pregunta) y las que hay con cada archivo."""
    with connect() as c:
        return c.execute("""
            SELECT m.conv, m.source_id, s.name, s.kind, max(m.id) AS last, count(*) AS n,
                   (SELECT content FROM messages f WHERE f.role = 'user' AND f.conv IS NOT DISTINCT FROM m.conv
                        AND f.source_id IS NOT DISTINCT FROM m.source_id ORDER BY f.id LIMIT 1) AS title
            FROM messages m LEFT JOIN sources s ON s.id = m.source_id
            GROUP BY m.conv, m.source_id, s.name, s.kind ORDER BY last DESC LIMIT 60""").fetchall()
