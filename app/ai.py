"""Modelos de IA de Multichat.

- Embeddings: **EmbeddingGemma-300M** servido por **llama.cpp** (`llama-server --embedding`, API compatible con
  OpenAI en /v1/embeddings). Si llama.cpp no está corriendo, usa el mismo modelo desde Ollama (`embeddinggemma`).
- Respuestas: modelo local en Ollama (qwen2.5:1.5b), un modelo de chat GGUF en llama.cpp, o Gemini por API (nube).
- Visión (imágenes de los PDF, fotos, fotogramas de video): moondream en Ollama o Gemini.
- Audio a texto: faster-whisper (local, CPU) o Gemini.
"""

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
from typing import AsyncIterator, Iterator

import logging

import httpx

TIMEOUT = httpx.Timeout(900.0, connect=10.0)
KEEP_ALIVE = "2m"   # libera la RAM del modelo a los 2 min sin uso (PC de 12 GB)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_OPENAI = GEMINI_BASE + "/openai/chat/completions"


log = logging.getLogger("multichat")


class AIError(Exception):
    pass


# ======================================================================= embeddings
# EmbeddingGemma se entrenó con estos prefijos: mejoran la búsqueda (pregunta corta vs. documento largo)
def query_prompt(q: str) -> str:
    return f"task: search result | query: {q}"


def doc_prompt(text: str, title: str = "none") -> str:
    return f"title: {title} | text: {text}"


def embed_label(s: dict) -> str:
    """Nombre del modelo de embeddings guardado junto a cada fuente (si cambia, hay que reindexar)."""
    return "embeddinggemma-300m"


def embed(texts: list[str], s: dict) -> list[list[float]]:
    if not texts:
        return []
    if os.getenv("MULTICHAT_TEST_EMBED") == "1":     # solo para pruebas automáticas sin modelos
        return [_hash_embed(t) for t in texts]
    errors = []
    order = ["llamacpp", "ollama"] if s["embed_provider"] == "llamacpp" else ["ollama", "llamacpp"]
    for prov in order:
        try:
            out = []
            for i in range(0, len(texts), 16):
                out += _embed_llamacpp(texts[i:i + 16], s) if prov == "llamacpp" else _embed_ollama(texts[i:i + 16], s)
            return out
        except AIError as e:
            errors.append(str(e))
    raise AIError(" / ".join(errors))


def _embed_llamacpp(texts: list[str], s: dict) -> list[list[float]]:
    url = s["llamacpp_embed_url"].rstrip("/") + "/v1/embeddings"
    try:
        r = httpx.post(url, json={"input": texts, "model": "embeddinggemma"}, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError(f"llama.cpp (embeddings) no responde en {s['llamacpp_embed_url']}") from e
    if r.status_code >= 400:
        raise AIError(f"llama.cpp respondió {r.status_code}: {r.text[:200]}")
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _embed_ollama(texts: list[str], s: dict) -> list[list[float]]:
    model = s["ollama_embed_model"]
    try:
        r = httpx.post(s["ollama_url"].rstrip("/") + "/api/embed",
                       json={"model": model, "input": texts, "keep_alive": KEEP_ALIVE}, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError(f"Ollama no responde en {s['ollama_url']}") from e
    if r.status_code == 404:
        raise AIError(f"Ollama no tiene '{model}' (ollama pull {model})")
    if r.status_code >= 400:
        raise AIError(f"Ollama respondió {r.status_code}: {r.text[:200]}")
    return r.json()["embeddings"]


def _hash_embed(text: str, dim: int = 768) -> list[float]:
    """Embedding de juguete (bolsa de palabras con hashing). Solo para las pruebas automáticas."""
    v = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        if len(w) < 3:
            continue
        h = int(hashlib.md5(w[:6].encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    return v


def embed_status(s: dict) -> dict:
    try:
        r = httpx.get(s["llamacpp_embed_url"].rstrip("/") + "/health", timeout=2)
        if r.status_code == 200:
            return {"ok": True, "via": "llama.cpp", "model": "EmbeddingGemma-300M (GGUF)"}
    except httpx.HTTPError:
        pass
    try:
        r = httpx.get(s["ollama_url"].rstrip("/") + "/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        if any(n.startswith(s["ollama_embed_model"]) for n in names):
            return {"ok": True, "via": "Ollama", "model": s["ollama_embed_model"],
                    "warning": "llama.cpp no está corriendo: se usa EmbeddingGemma desde Ollama."}
    except (httpx.HTTPError, ValueError):
        pass
    if os.getenv("MULTICHAT_TEST_EMBED") == "1":
        return {"ok": True, "via": "prueba", "model": "hash"}
    return {"ok": False, "error": "No hay modelo de embeddings: inicia llama.cpp (iniciar.bat lo hace solo) "
                                  "o descarga 'ollama pull embeddinggemma'."}


# ======================================================================= chat
def _openai_stream(url: str, headers: dict, body: dict) -> Iterator[str]:
    with httpx.Client(timeout=TIMEOUT) as c:
        try:
            with c.stream("POST", url, json=body, headers=headers) as r:
                if r.status_code >= 400:
                    r.read()
                    raise AIError(f"{url.split('/')[2]} respondió {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        piece = json.loads(data)["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, ValueError):
                        continue
                    if piece:
                        yield piece
        except httpx.HTTPError as e:
            raise AIError(f"No me pude conectar a {url.split('/')[2]}: {e}") from e


def chat_stream(messages: list[dict], s: dict, max_tokens: int = 600) -> Iterator[str]:
    prov = s["chat_provider"]
    if os.getenv("MULTICHAT_TEST_EMBED") == "1" and prov == "ollama":
        yield "(respuesta de prueba) " + messages[-1]["content"][-200:]
        return
    if prov == "gemini":
        key = s.get("gemini_api_key", "").strip()
        if not key:
            raise AIError("Falta la clave de Gemini en ⚙ Ajustes.")
        yield from _openai_stream(GEMINI_OPENAI, {"Authorization": f"Bearer {key}"},
                                  {"model": s["gemini_model"], "messages": messages, "stream": True,
                                   "temperature": 0.2, "max_tokens": max_tokens})
        return
    if prov == "llamacpp":
        yield from _openai_stream(s["llamacpp_chat_url"].rstrip("/") + "/v1/chat/completions", {},
                                  {"messages": messages, "stream": True, "temperature": 0.2, "max_tokens": max_tokens})
        return
    body = {"model": s["chat_model"], "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": max_tokens}}
    with httpx.Client(timeout=TIMEOUT) as c:
        try:
            with c.stream("POST", s["ollama_url"].rstrip("/") + "/api/chat", json=body) as r:
                if r.status_code == 404:
                    raise AIError(f"Ollama no tiene el modelo '{s['chat_model']}' (ollama pull {s['chat_model']})")
                if r.status_code >= 400:
                    r.read()
                    raise AIError(f"Ollama respondió {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("error"):
                            raise AIError(obj["error"])
                        piece = obj.get("message", {}).get("content", "")
                        if piece:
                            yield piece
        except httpx.ConnectError as e:
            raise AIError(f"No me pude conectar a Ollama en {s['ollama_url']}. Ábrelo desde el menú Inicio.") from e


def chat_once(messages: list[dict], s: dict, max_tokens: int = 400) -> str:
    return "".join(chat_stream(messages, s, max_tokens)).strip()


# ======================================================================= visión
VISION_PROMPT_EN = "Describe this image in detail: objects, people, colors, any visible text, charts or diagrams."
VISION_PROMPT_ES = ("Describe esta imagen en español, en 3 a 5 frases: qué objetos, personas, colores, gráficos o "
                    "diagramas aparecen y cualquier texto visible.")


def describe_image(png: bytes, s: dict) -> str:
    """Descripción en español de una imagen (para poder buscarla y preguntar por ella)."""
    prov = s["vision_provider"]
    if prov == "no":
        return ""
    if os.getenv("MULTICHAT_TEST_EMBED") == "1" and prov == "ollama":
        return "imagen de prueba"
    png = _shrink(png)
    b64 = base64.b64encode(png).decode()
    if prov == "gemini":
        key = s.get("gemini_api_key", "").strip()
        if not key:
            raise AIError("Falta la clave de Gemini en ⚙ Ajustes (o cambia la visión a Local).")
        body = {"model": s["gemini_model"], "temperature": 0.1, "max_tokens": 400, "messages": [{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT_ES},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
        r = httpx.post(GEMINI_OPENAI, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
        if r.status_code >= 400:
            raise AIError(f"Gemini respondió {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"]["content"] or "").strip()
    body = {"model": s["vision_model"], "stream": False, "keep_alive": "1m",
            "options": {"temperature": 0.1, "num_ctx": 2048, "num_predict": 180},
            "messages": [{"role": "user", "content": VISION_PROMPT_EN, "images": [b64]}]}
    try:
        r = httpx.post(s["ollama_url"].rstrip("/") + "/api/chat", json=body, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError("Ollama no responde (visión).") from e
    if r.status_code == 404:
        raise AIError(f"Ollama no tiene '{s['vision_model']}' (ollama pull {s['vision_model']})")
    english = r.json()["message"]["content"].strip()
    if not english:
        return ""
    # moondream solo habla inglés: se traduce con el modelo de chat para que la respuesta salga en español
    try:
        es = chat_once([{"role": "system", "content": "Traduce al español. Responde solo con la traducción."},
                        {"role": "user", "content": english}], s, max_tokens=300)
        return f"{es}\n(EN: {english})" if es else english
    except AIError:
        return english


def _shrink(png: bytes, side: int = 768) -> bytes:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png))
        if max(img.size) <= side:
            return png
        img.thumbnail((side, side))
        out = io.BytesIO()
        img.convert("RGB").save(out, "PNG")
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return png


# ======================================================================= audio → texto
_whisper = {}


def transcribe(path: str, s: dict) -> tuple[list[dict], float]:
    """[(start, end, text)] y duración en segundos. Sirve para audios y para la pista de audio de los videos."""
    if s["transcribe_provider"] == "gemini":
        return _transcribe_gemini(path, s)
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise AIError("Falta faster-whisper (pip install -r requirements.txt)") from e
    name = s.get("whisper_model", "base")
    try:
        if name not in _whisper:
            _whisper.clear()
            _whisper[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2)
        audio = _load_audio(path)
        try:
            segs, info = _run_whisper(_whisper[name], audio, vad=True)
        except Exception as e:  # noqa: BLE001  (el filtro de silencios usa onnxruntime; si falla, se intenta sin él)
            log.warning("Whisper con filtro de silencios falló (%s); reintento sin filtro", e)
            segs, info = _run_whisper(_whisper[name], audio, vad=False)
        if not segs:   # con el filtro a veces se 'come' audios suaves: segundo intento sin filtro
            segs, info = _run_whisper(_whisper[name], audio, vad=False)
    except Exception as e:  # noqa: BLE001
        log.exception("faster-whisper falló")
        if s.get("gemini_api_key", "").strip():
            log.warning("Uso Gemini para transcribir porque Whisper falló")
            return _transcribe_gemini(path, s)
        raise AIError(f"Whisper no pudo transcribir ({type(e).__name__}: {e}). "
                      "Si dice DLL, instala 'Microsoft Visual C++ Redistributable x64' o usa Gemini en ⚙ Ajustes.") from e
    return segs, float(info.duration or (segs[-1]["end"] if segs else 0))


def _load_audio(path: str):
    """Decodifica cualquier audio/video a 16 kHz mono con PyAV (sirve con todas sus versiones;
    el lector propio de faster-whisper falla con PyAV 15+ por el parámetro 'metadata_errors')."""
    import av
    import numpy as np
    parts = []
    with av.open(path) as c:
        st = next((x for x in c.streams if x.type == "audio"), None)
        if st is None:
            raise AIError("El archivo no tiene pista de audio.")
        rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in c.decode(st):
            for f in rs.resample(frame) or []:
                parts.append(f.to_ndarray().reshape(-1))
        for f in rs.resample(None) or []:
            parts.append(f.to_ndarray().reshape(-1))
    if not parts:
        raise AIError("El audio está vacío.")
    return np.concatenate(parts).astype(np.float32) / 32768.0


def _run_whisper(model, path, vad: bool):
    audio = _load_audio(path) if isinstance(path, str) else path
    segments, info = model.transcribe(audio, language="es", vad_filter=vad, beam_size=1)
    segs = [{"start": float(x.start), "end": float(x.end), "text": x.text.strip()} for x in segments if x.text.strip()]
    return segs, info


def _transcribe_gemini(path: str, s: dict) -> tuple[list[dict], float]:
    key = s.get("gemini_api_key", "").strip()
    if not key:
        raise AIError("Falta la clave de Gemini en ⚙ Ajustes (o cambia la transcripción a Local).")
    data = open(path, "rb").read()
    if len(data) > 18 * 1024 * 1024:
        raise AIError("Archivo muy grande para enviarlo a Gemini (máx. 18 MB); usa la transcripción local.")
    mime = mimetypes.guess_type(path)[0] or "audio/mpeg"
    prompt = ("Transcribe este audio en español. Responde SOLO con JSON: una lista de objetos "
              '{"start": segundos, "end": segundos, "text": "..."} con frases de 5 a 15 segundos.')
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
                                    {"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    r = httpx.post(f"{GEMINI_BASE}/models/{s['gemini_model']}:generateContent", params={"key": key},
                   json=body, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise AIError(f"Gemini respondió {r.status_code}: {r.text[:200]}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    segs = [{"start": float(x.get("start", 0)), "end": float(x.get("end", 0)), "text": str(x.get("text", "")).strip()}
            for x in json.loads(text) if str(x.get("text", "")).strip()]
    return segs, segs[-1]["end"] if segs else 0.0


def whisper_ok() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def ollama_models(s: dict) -> list[str]:
    try:
        r = httpx.get(s["ollama_url"].rstrip("/") + "/api/tags", timeout=2)
        return sorted({m["name"].removesuffix(":latest") for m in r.json().get("models", [])})
    except (httpx.HTTPError, ValueError):
        return []
