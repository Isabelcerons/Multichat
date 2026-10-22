"""Configuración de Multichat. Los valores se leen del archivo .env y se pueden cambiar desde ⚙ Ajustes."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_env = BASE_DIR / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------- Base de datos: PostgreSQL ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/multichat")

# ---------- Base de conocimiento: carpeta con PDF, audios, videos e imágenes ----------
KB_DIR = Path(os.getenv("KB_DIR", BASE_DIR / "conocimiento"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
IMG_DIR = DATA_DIR / "imagenes"          # imágenes sacadas de los PDF y fotogramas de los videos

PDF_EXT = {".pdf"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".m4b", ".ogg", ".oga", ".flac", ".aac", ".opus", ".amr", ".wma", ".weba", ".aiff", ".aif", ".caf", ".mp2", ".mpga"}
VIDEO_EXT = {".mp4", ".m4v", ".3gp", ".mkv", ".mov", ".webm", ".avi", ".wmv", ".flv", ".mpeg", ".mpg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MAX_PAGES = int(os.getenv("MAX_PAGES", "150"))
MAX_IMAGES_PER_PDF = int(os.getenv("MAX_IMAGES_PER_PDF", "20"))
MIN_IMAGE_PX = int(os.getenv("MIN_IMAGE_PX", "120"))      # ignora íconos y adornos
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
AUDIO_WINDOW_S = 30          # la transcripción se agrupa en trozos de ~30 segundos
TOP_K = 5
EMBED_DIM = 768              # EmbeddingGemma-300M produce vectores de 768 números

DEFAULT_SETTINGS = {
    # Embeddings (búsqueda semántica): EmbeddingGemma servido por llama.cpp (llama-server --embedding)
    "embed_provider": os.getenv("EMBED_PROVIDER", "llamacpp"),        # llamacpp | ollama
    "llamacpp_embed_url": os.getenv("LLAMACPP_EMBED_URL", "http://localhost:8081"),
    "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
    # Respuestas: modelo local (Ollama), llama.cpp (si cargas un modelo de chat GGUF) o nube (Gemini)
    "chat_provider": os.getenv("CHAT_PROVIDER", "ollama"),            # ollama | llamacpp | gemini
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "chat_model": os.getenv("CHAT_MODEL", "qwen2.5:1.5b"),
    "llamacpp_chat_url": os.getenv("LLAMACPP_CHAT_URL", "http://localhost:8082"),
    # Mirar imágenes (PDF, fotos y fotogramas de video)
    "vision_provider": os.getenv("VISION_PROVIDER", "ollama"),        # ollama | gemini | no
    "vision_model": os.getenv("VISION_MODEL", "moondream"),
    # Pasar audio a texto: faster-whisper (local) o Gemini (nube)
    "transcribe_provider": os.getenv("TRANSCRIBE_PROVIDER", "local"),  # local | gemini
    "whisper_model": os.getenv("WHISPER_MODEL", "base"),               # tiny | base | small
    "video_frames": os.getenv("VIDEO_FRAMES", "12"),                   # máximo de escenas distintas a describir por video
    # Nube
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    # OCR
    "ocr_lang": os.getenv("OCR_LANG", "spa+eng"),
    "tesseract_cmd": os.getenv("TESSERACT_CMD", ""),
}

SECRET_KEYS = {"gemini_api_key"}


def _only_audio(path: Path) -> bool:
    """Algunos 'videos' son solo audio (p. ej. los audios de WhatsApp Web se descargan como .mpeg)."""
    try:
        if not path.is_file():
            return False
        import av
        with av.open(str(path)) as c:
            return not any(st.type == "video" for st in c.streams) and any(st.type == "audio" for st in c.streams)
    except Exception:  # noqa: BLE001
        return False


def kind_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return "pdf"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "audio" if _only_audio(path) else "video"
    if ext in IMAGE_EXT:
        return "imagen"
    return None
