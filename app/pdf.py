"""Lectura de PDF con PyMuPDF: texto por página, imágenes, páginas escaneadas y render con resaltado."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

SCANNED_TEXT_LIMIT = 40      # una página con menos caracteres se trata como escaneada
MAX_IMAGE_SIDE = 1600


@dataclass
class PageImage:
    page: int
    png: bytes
    width: int
    height: int


@dataclass
class ParsedPDF:
    total_pages: int
    pages: list[tuple[int, str]] = field(default_factory=list)     # texto real del PDF
    scanned: list[PageImage] = field(default_factory=list)          # páginas sin texto (para OCR)
    images: list[PageImage] = field(default_factory=list)           # imágenes incrustadas
    skipped_images: int = 0


def clean_text(text: str) -> str:
    text = text.replace("­", "")                       # guiones suaves
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)          # palabras cortadas al final de línea
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _png_from_xref(doc, xref: int) -> tuple[bytes, int, int] | None:
    try:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4 or pix.colorspace is None or pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        while max(pix.width, pix.height) > MAX_IMAGE_SIDE:
            pix.shrink(1)
        return pix.tobytes("png"), pix.width, pix.height
    except Exception:
        return None


def parse(data: bytes, max_pages: int, max_images: int, min_px: int) -> ParsedPDF:
    doc = pymupdf.open(stream=data, filetype="pdf")
    out = ParsedPDF(total_pages=doc.page_count)
    seen: set[int] = set()
    for i in range(min(doc.page_count, max_pages)):
        page, page_no = doc[i], i + 1
        text = clean_text(page.get_text("text"))
        out.pages.append((page_no, text))
        if len(text) < SCANNED_TEXT_LIMIT and (page.get_images() or page.get_drawings()):
            pix = page.get_pixmap(dpi=200)  # 200 dpi: buena lectura para Tesseract
            out.scanned.append(PageImage(page_no, pix.tobytes("png"), pix.width, pix.height))
            continue
        for img in page.get_images(full=True):
            xref, w, h = img[0], img[2], img[3]
            if xref in seen or w < min_px or h < min_px:
                continue
            seen.add(xref)
            if len(out.images) >= max_images:
                out.skipped_images += 1
                continue
            res = _png_from_xref(doc, xref)
            if res:
                out.images.append(PageImage(page_no, *res))
    doc.close()
    return out


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[tuple[int, str]]:
    """Corta cada página en trozos de ~`size` caracteres con `overlap` de solapamiento,
    cortando preferiblemente en fin de frase y sin partir palabras."""
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = max(window.rfind(". "), window.rfind("\n"))
                if cut > size * 0.5:
                    end = start + cut + 1
                else:
                    sp = window.rfind(" ")
                    if sp > size * 0.5:
                        end = start + sp
            piece = text[start:end].strip()
            if len(piece) > 20:
                chunks.append((page_no, piece))
            if end >= len(text):
                break
            nxt = end - overlap
            if nxt > start:
                sp = text.find(" ", nxt, end)
                start = sp + 1 if sp != -1 else nxt
            else:
                start = end
    return chunks


def render_page(path: str | Path, page_no: int, highlight: str | None = None, dpi: int = 110,
                words: list[str] | None = None) -> bytes:
    """PNG de una página; si se da `highlight`, resalta en amarillo donde aparece ese texto, y cada palabra de
    `words` (las que buscó el usuario) en naranja fuerte para encontrarla de un vistazo."""
    doc = pymupdf.open(path)
    try:
        page = doc[page_no - 1]
        want = {w.lower() for w in words or []}
        for x0, y0, x1, y1, word, *_ in (page.get_text("words") if want else []):   # solo palabras completas
            if word.strip(".,;:¿?¡!()«»\"'").lower() in want:
                a = page.add_highlight_annot(pymupdf.Rect(x0, y0, x1, y1))
                a.set_colors(stroke=(1, 0.55, 0.1))
                a.update()
        if highlight:
            rects = page.search_for(highlight)
            if not rects:  # intenta con las primeras palabras si la frase completa no aparece tal cual
                words = highlight.split()
                for n in (8, 5, 3):
                    if len(words) > n:
                        rects = page.search_for(" ".join(words[:n]))
                        if rects:
                            break
            for r in rects:
                page.add_highlight_annot(r)
        return page.get_pixmap(dpi=dpi, annots=True).tobytes("png")
    finally:
        doc.close()
