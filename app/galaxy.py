"""'Galaxia del conocimiento': proyecta los embeddings de 768 dimensiones a 2D (PCA) para dibujarlos.

Fragmentos con significado parecido quedan cerca en el mapa. La pregunta se proyecta con la misma base,
así se ve en qué 'constelación' cae y cuáles son sus vecinos más cercanos."""

import numpy as np

from . import ai, db, indexer

_cache: dict = {"key": None}


def _build(s: dict) -> dict:
    model = ai.embed_label(s)
    rows = db.map_rows(model)
    key = (model, len(rows), rows[-1]["id"] if rows else 0)
    if _cache["key"] == key:
        return _cache
    if len(rows) < 3:
        _cache.update(key=key, points=[], mean=None, basis=None, scale=1.0)
        return _cache
    X = np.array([r["emb"] for r in rows], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    basis = vt[:2].T                                   # 768 x 2
    P = (X - mean) @ basis
    scale = float(np.abs(P).max()) or 1.0
    P /= scale
    points = []
    for r, (x, y) in zip(rows, P):
        where = f"pág. {r['page']}" if r["page"] else (indexer.fmt_time(r["t_start"]) if r["t_start"] is not None else "")
        points.append({"id": r["id"], "x": round(float(x), 4), "y": round(float(y), 4), "kind": r["kind"],
                       "source_kind": r["source_kind"], "source_id": r["source_id"], "name": r["name"],
                       "where": where, "text": r["content"]})
    _cache.update(key=key, points=points, mean=mean, basis=basis, scale=scale)
    return _cache


def points(s: dict) -> list[dict]:
    return _build(s)["points"]


def project(vec: list[float], s: dict) -> list[float] | None:
    g = _build(s)
    if g.get("basis") is None:
        return None
    v = np.asarray(vec, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    x, y = ((v - g["mean"]) @ g["basis"]) / g["scale"]
    return [round(float(np.clip(x, -1.2, 1.2)), 4), round(float(np.clip(y, -1.2, 1.2)), 4)]


def trace_event(trace: dict, hits: list[dict], s: dict) -> dict:
    """Lo que muestra el 'modo cerebro': vector de la pregunta, tiempos y posición en la galaxia."""
    q = trace.get("qvec") or []
    try:
        xy = project(q, s) if q else None
    except Exception:  # noqa: BLE001
        xy = None
    return {"type": "trace", "dim": len(q), "vector": [round(float(x), 3) for x in q[:14]],
            "embed_ms": trace.get("embed_ms"), "search_ms": trace.get("search_ms"),
            "compared": db.stats()["chunks"], "query_xy": xy, "hit_ids": [h["id"] for h in hits]}
