"""Consulta semántica (RAG): pregunta → embedding (EmbeddingGemma) → fragmentos más parecidos en PostgreSQL
(de PDF, imágenes, audios y videos) → el modelo responde usando solo esos fragmentos y citando la fuente."""

import re
import unicodedata

from . import ai, db
from .config import TOP_K
from .indexer import fmt_time

STOP = set("""a al algo como con de del el en es esta este esto la las lo los mas me mi o para pero por que se si sin
sobre su sus tambien un una uno unos unas y ya hay cual cuales donde cuando quien que dice dicen habla hablan
explica explican video videos audio audios pdf documento documentos imagen imagenes archivo archivos base
conocimiento sobre segun tiene tienen minuto minutos segundo segundos momento parte menciona mencionan""".split())

SYSTEM = """Eres Multichat, un asistente que responde preguntas sobre una base de conocimiento con documentos PDF,
imágenes, audios y videos. Reglas:
1. Responde SIEMPRE en español, claro y breve (máximo 8 frases o una lista corta).
2. Usa SOLO la información de los FRAGMENTOS. Si no está, di: "No encontré esa información en la base de conocimiento."
3. Después de cada dato, cita la fuente tal como aparece en el fragmento, por ejemplo [manual.pdf, pág. 3] o
   [clase.mp3, 01:20].
   En audios y videos cada frase trae su marca [mm:ss]: di en qué minuto y segundo se dice, por ejemplo
   "lo dice en el minuto 00:38".
4. Las imágenes y escenas de video vienen descritas en texto: puedes responder sobre lo que se ve en ellas.
5. Los audios y videos se transcribieron automáticamente y algunas palabras pueden estar mal escritas (por ejemplo
   «lapisero» en vez de «lapicero» o lápiz, «Padcas» en vez de «podcast»). Interprétalas por cómo suenan."""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def keywords(q: str) -> list[str]:
    return [w for w in re.findall(r"\w+", norm(q)) if len(w) > 2 and w not in STOP][:8]


def fuzzy(w: str) -> str:
    """Expresión regular que tolera los errores típicos de Whisper en español (letras que suenan igual):
    s/z/c, b/v, y/ll, h muda. Así 'lapiz' encuentra 'lapisero' y 'lapicero'. Sirve en Python y en PostgreSQL."""
    w, out, i = w[:7], [], 0
    while i < len(w):
        ch = w[i]
        if w.startswith("ll", i):
            out.append("(?:ll|y)"); i += 2; continue
        out.append({"s": "[szc]", "z": "[szc]", "c": "[cszk]", "k": "[ck]", "b": "[bv]", "v": "[bv]",
                    "y": "(?:ll|y)", "h": "h?"}.get(ch, ch))
        i += 1
    return "".join(out)


def cite(r: dict) -> str:
    if r.get("page"):
        return f"{r['name']}, pág. {r['page']}"
    if r.get("t_start") is not None:
        return f"{r['name']}, {fmt_time(r['t_start'])}"
    return r["name"]


def timeline(h: dict) -> list[dict]:
    """Las frases de una transcripción con su segundo exacto. Los fragmentos indexados antes de guardar los tiempos
    solo tienen el inicio y el fin del trozo: ahí el segundo de cada frase se estima por su posición en el texto."""
    if h["kind"] == "texto":                   # PDF: sus frases, sin tiempo, para citarlas textualmente
        body = re.sub(r"[ \t]*\n(?!\s*\n)[ \t]*", " ", h["content"])      # une las líneas de un mismo párrafo
        parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", body)
        return [{"start": None, "text": x.strip()} for x in parts if len(x.strip()) >= 12]
    if h.get("segments"):
        return h["segments"]
    body = h["content"].split("]: ", 1)[-1]
    t0, t1 = h.get("t_start") or 0.0, h.get("t_end") or h.get("t_start") or 0.0
    out, pos = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", body):
        i = body.find(sent, pos)
        out.append({"start": round(t0 + (t1 - t0) * i / max(len(body), 1), 1), "text": sent.strip()})
        pos = i + len(sent)
    return [x for x in out if x["text"]]


def locate(question: str, hits: list[dict], s: dict, qvec: list[float] | None = None) -> None:
    """Para cada fragmento de texto, busca la frase exacta que responde la pregunta: en audios y videos guarda su
    minuto y segundo (t_exact); en los PDF, la frase para citarla textualmente con su página. Puntaje de cada frase =
    palabras clave (tolerando errores de Whisper; una palabra que está en muchas frases vale menos) + similitud de
    significado con EmbeddingGemma (esta solo en los 3 fragmentos más parecidos, para no hacer lenta la respuesta)."""
    import numpy as np
    regs = [re.compile(fuzzy(w)) for w in keywords(question)]
    trans = sorted((h for h in hits if h["kind"] in ("transcripcion", "texto") and timeline(h)),
                   key=lambda h: -(h.get("score") or 0))
    if not trans:
        return
    flat = [(h, x) for h in trans[:3] for x in timeline(h)]
    sims = np.zeros(len(flat), dtype=np.float32)
    if flat:
        q = np.asarray(qvec or ai.embed([ai.query_prompt(question)], s)[0], dtype=np.float32)
        vecs = np.asarray(ai.embed([ai.doc_prompt(x["text"]) for _, x in flat], s), dtype=np.float32)
        sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    sim_of = {id(x): float(v) for (_, x), v in zip(flat, sims)}
    for h in trans:
        segs = timeline(h)
        found = [[bool(r.search(norm(x["text"]))) for r in regs] for x in segs]
        df = [max(1, sum(f[j] for f in found)) for j in range(len(regs))]
        score = [sum(1 / df[j] for j in range(len(regs)) if f[j]) + 2 * sim_of.get(id(x), 0.0)
                 for x, f in zip(segs, found)]
        if max(score) > 0:
            i = score.index(max(score))
            seg = segs[i]
            if h["kind"] == "texto" and seg["text"].endswith(("?", ":")) and i + 1 < len(segs):
                seg = {"start": None, "text": f"{seg['text']} {segs[i + 1]['text']}"}      # título o pregunta: + la respuesta
            _pin(h, seg)
            h["marks"] = marks(question, h["exact"])
            h["exact"] = _trim(h["exact"], h["marks"])
    for h in hits:                  # las palabras buscadas también en los demás fragmentos (para resaltarlas)
        if "marks" not in h and h["kind"] in ("texto", "transcripcion"):
            h["marks"] = marks(question, h["content"])


def marks(question: str, text: str, limit: int = 6) -> list[str]:
    """Las palabras del texto que corresponden a las de la pregunta, tal como están escritas en el documento
    (p. ej. 'lapisero' si preguntaron por 'lápiz'): la página las usa para resaltarlas en el chat y en el PDF."""
    regs = [re.compile(fuzzy(w)) for w in keywords(question)]
    out = []
    for tok in re.findall(r"\w+", text):
        if len(tok) > 2 and tok not in out and any(r.match(norm(tok)) for r in regs):
            out.append(tok)
            if len(out) >= limit:
                break
    return out


def _trim(text: str, words: list[str], size: int = 120) -> str:
    """Una cita larga se recorta alrededor de la primera palabra buscada, para no citar texto de más."""
    if len(text) <= size:
        return text
    pos = min((m.start() for w in words for m in [re.search(re.escape(w), text)] if m), default=0)
    a = max(0, pos - size // 3)
    a = text.rfind(" ", 0, a) + 1 if a else 0
    b = text.find(" ", min(len(text), a + size))
    b = len(text) if b < 0 else b
    return ("…" if a else "") + text[a:b].strip() + ("…" if b < len(text) else "")


def _pin(h: dict, seg: dict) -> None:
    h["exact"] = " ".join(seg["text"].split())
    if seg.get("start") is not None:          # audio o video: la cita lleva el minuto y segundo exactos
        h["t_exact"] = seg["start"]
        h["cite"] = f"{h['name']}, {fmt_time(seg['start'])}"


def retrieve(question: str, s: dict, kinds: list[str] | None = None, k: int = TOP_K,
             trace: dict | None = None, source_id: int | None = None, only_relevant: bool = True) -> list[dict]:
    """Búsqueda híbrida: similitud semántica + un pequeño refuerzo si el fragmento contiene las palabras clave.
    Si se pasa `trace`, se llena con el vector de la pregunta y los tiempos (para el 'modo cerebro')."""
    import time
    t0 = time.time()
    qvec = ai.embed([ai.query_prompt(question)], s)[0]
    t1 = time.time()
    cands = {r["id"]: r for r in db.nearest(qvec, ai.embed_label(s), kinds, limit=k * 4, source_id=source_id)}
    if trace is not None:
        trace.update(qvec=qvec, embed_ms=round((t1 - t0) * 1000), search_ms=round((time.time() - t1) * 1000))
    words = keywords(question)
    if words:
        for r in db.keyword_hits([fuzzy(w) for w in words], kinds, limit=30, source_id=source_id):
            if r["id"] not in cands:
                r["score"] = 0.0
                cands[r["id"]] = r
    for r in cands.values():
        text = norm(r["content"])
        hits = r["kw"] = sum(1 for w in words if re.search(fuzzy(w), text))
        r["semantic"] = float(r.get("score") or 0.0)
        r["score"] = r["semantic"] + 0.06 * hits
    if any(r["semantic"] == 0.0 for r in cands.values()):
        # fragmentos que llegaron solo por palabra clave: calcula también su similitud semántica
        missing = [r for r in cands.values() if r["semantic"] == 0.0]
        vecs = ai.embed([ai.doc_prompt(r["content"]) for r in missing], s)
        import numpy as np
        q = np.asarray(qvec, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        for r, v in zip(missing, vecs):
            v = np.asarray(v, dtype=np.float32)
            r["semantic"] = float(v @ q / (np.linalg.norm(v) + 1e-9))
            r["score"] += r["semantic"]
    ranked = sorted(cands.values(), key=lambda r: -r["score"])
    out, per_source = [], {}
    top = ranked[0]["score"] if ranked else 0.0
    for r in ranked:                       # máximo 2 fragmentos por archivo (más variedad), salvo que sea casi tan bueno como el mejor
        if not source_id and per_source.get(r["source_id"], 0) >= 2 and r["score"] < 0.85 * top:
            continue
        # solo lo relevante: el mejor siempre; los demás si se acercan a él (60 %), o al 50 % si tienen alguna
        # palabra de la pregunta. Así no se rellena con fragmentos que no tienen nada que ver.
        if only_relevant and out and r["score"] < (0.5 if r["kw"] else 0.6) * top:
            continue
        per_source[r["source_id"]] = per_source.get(r["source_id"], 0) + 1
        r["cite"] = cite(r)
        out.append(r)
        if len(out) >= k:
            break
    return out


VISUAL_WORDS = ("ve", "ven", "vea", "imagen", "imagenes", "foto", "fotos", "color", "aparece", "aparecen", "muestra", "muestran",
                "escena", "escenas", "dibujo", "grafica", "objeto", "persona", "visual", "insertan", "salen")
TIPO = {"audio": "un audio", "video": "un video", "pdf": "un documento PDF", "imagen": "una imagen"}


def retrieve_file(question: str, s: dict, source_id: int, trace: dict | None = None, budget: int = 3600) -> list[dict]:
    """Para 'hablar con un archivo': le da al modelo TODO el texto del archivo en orden (si cabe),
    más las descripciones visuales más relacionadas con la pregunta. Así un modelo pequeño entiende el contexto completo."""
    ranked = retrieve(question, s, None, k=40, trace=trace, source_id=source_id, only_relevant=False)
    score = {r["id"]: r["score"] for r in ranked}
    allc = db.source_chunks(source_id)
    for c in allc:
        c["score"] = score.get(c["id"], 0.0)
        c["semantic"] = c["score"]
        c["cite"] = cite(c)
    text = [c for c in allc if c["kind"] in ("texto", "transcripcion")]
    visual = [c for c in allc if c["kind"] in ("imagen", "escena")]
    wants_visual = any(w in re.findall(r"\w+", norm(question)) for w in VISUAL_WORDS) or not text
    if wants_visual and visual:      # preguntan qué se ve: menos texto, para que quepan todas las escenas
        budget = min(budget, 1500)
    if sum(len(c["content"]) for c in text) > budget:      # no cabe todo: los más parecidos, luego en orden
        keep, used = set(), 0
        for c in sorted(text, key=lambda c: -c["score"]):
            if used + len(c["content"]) > budget:
                continue
            keep.add(c["id"]); used += len(c["content"])
        text = [c for c in text if c["id"] in keep]
    visual = sorted(visual, key=lambda c: -c["score"])[: (12 if wants_visual else 2)]      # preguntan qué se ve: todas
    visual.sort(key=lambda c: (c["page"] or 0, c["t_start"] or 0))
    return text + visual


def key_sentences(question: str, hits: list[dict], limit: int = 4) -> str:
    """Las frases de los fragmentos que contienen las palabras de la pregunta (con tolerancia a errores de Whisper,
    p. ej. 'lapiz' → 'lapisero'), para ponerlas justo después de la pregunta. Un modelo pequeño (qwen2.5:1.5b) se
    pierde en un contexto largo y no relaciona solo 'lápiz' con 'lapisero'; así ve primero lo importante."""
    regs = [re.compile(fuzzy(w)) for w in keywords(question)]
    scored = []
    for rank, h in enumerate(hits):
        if h["kind"] == "transcripcion":         # cada frase con su minuto y segundo exactos
            sents = [(f"{h['name']}, {fmt_time(x['start'])}", x["text"]) for x in timeline(h)]
        elif h["kind"] == "texto":
            sents = [(h["cite"], x) for x in re.split(r"(?<=[.!?])\s+", h["content"])]
        else:
            continue
        for where, sent in sents:
            n = sum(1 for r in regs if r.search(norm(sent)))
            if n:
                scored.append((-n, rank, f"- [{where}] {sent.strip()}"))
    if not scored:
        return ""
    return "FRASES MÁS RELACIONADAS CON LA PREGUNTA:\n" + "\n".join(s for *_, s in sorted(scored)[:limit])


def context(h: dict) -> str:
    """Un fragmento como lo ve el modelo. En audios y videos va frase por frase con su [archivo, mm:ss]."""
    if h["kind"] == "transcripcion":
        lines = "\n".join(f"[{h['name']}, {fmt_time(x['start'])}] {x['text']}" for x in timeline(h))
        return f"(lo que se dice en {h['name']})\n{lines}"
    if h["kind"] in ("imagen", "escena"):
        return f"[{h['cite']}] ({'lo que se ve' if h['kind'] == 'escena' else 'imagen'})\n{visual_text(h)}"
    return f"[{h['cite']}] ({h['kind']})\n{h['content']}"


def visual_text(h: dict, limit: int = 420) -> str:
    """La descripción de una imagen o escena sin la copia en inglés de moondream (la duplica y llena el contexto
    del modelo pequeño, que solo tiene 4096 tokens) y recortada a lo esencial."""
    text = re.sub(r"\s*\(EN:.*?\)(?=\.?\s*(Texto visible|$))", "", h["content"], flags=re.S)
    text = re.sub(r"^(Escena del video en el minuto [\d:]+|Imagen[^.]*)\.\s*Descripción:\s*", "", text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def scene_list(question: str, hits: list[dict]) -> str:
    """Si preguntan qué imágenes o escenas hay, la lista de lo que se ve con su minuto, justo después de la pregunta."""
    if not any(w in re.findall(r"\w+", norm(question)) for w in VISUAL_WORDS):
        return ""
    vis = sorted((h for h in hits if h["kind"] in ("escena", "imagen")),
                 key=lambda h: (h.get("page") or 0, h.get("t_start") or 0))
    if not vis:
        return ""
    first = lambda h: re.split(r"(?<=[.!?])\s", visual_text(h, 300))[0]  # noqa: E731
    return "LO QUE SE VE (escenas e imágenes, en orden):\n" + "\n".join(f"- [{h['cite']}] {first(h)}" for h in vis)


def build_messages(question: str, hits: list[dict], history: list[dict], target: dict | None = None) -> list[dict]:
    ctx = "\n\n".join(context(h) for h in hits)
    key = scene_list(question, hits) or key_sentences(question, hits)
    question = f"{question}\n\n{key}" if key else question      # después de la pregunta: así el modelo pequeño acierta más
    if target:      # conversación con un solo archivo: sin historial de otros archivos, instrucciones más directas
        system = SYSTEM + f"""
6. El usuario está hablando con {TIPO.get(target['kind'], 'un archivo')} llamado «{target['name']}». Los FRAGMENTOS son su contenido
   completo en orden{' (lo que se dice, con su minuto, y lo que se ve en algunas escenas)' if target['kind'] in ('audio', 'video') else ''}.
   Si preguntan de qué trata, resume la idea principal. Responde la pregunta directamente usando ese contenido."""
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"CONTENIDO DE «{target['name']}»:\n{ctx}\n\nPREGUNTA: {question}"}]
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in history[-4:]:
        msgs.append({"role": m["role"], "content": m["content"][:600]})
    msgs.append({"role": "user", "content": f"FRAGMENTOS:\n{ctx}\n\nPREGUNTA: {question}"})
    return msgs


def public(h: dict) -> dict:
    """Lo que ve la página web de cada fuente encontrada."""
    return {"chunk_id": h["id"], "source_id": h["source_id"], "name": h["name"], "source_kind": h["source_kind"],
            "kind": h["kind"], "page": h.get("page"), "t_start": h.get("t_exact", h.get("t_start")), "cite": h["cite"],
            "image": bool(h.get("image_path")), "score": round(float(h.get("semantic") or 0), 3),
            "match": round(float(h.get("score") or 0), 3),      # similitud + refuerzo por palabras clave (el orden final)
            "exact": h.get("exact"), "marks": h.get("marks", []), "text": (f"«{h['exact']}» " if h.get("exact") else "") + h["content"][:420]}
