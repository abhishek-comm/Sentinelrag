import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document as DocxDocument

from .embeddings import local_embed
from .storage import connection, utcnow

INJECTION_PATTERNS = (
    "ignore previous instructions", "ignore all previous", "system message", "developer message",
    "reveal your prompt", "do not cite", "you are chatgpt",
)


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def sanitize_source_text(text: str) -> str:
    # Source files can contain ordinary content and a hostile instruction on one line.
    # Remove only the malicious sentence, preserving nearby document evidence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    safe = [sentence for sentence in sentences if not any(pattern in sentence.lower()
                                                          for pattern in INJECTION_PATTERNS)]
    return " ".join(safe)


def extract_pages(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            pages = [PageText(i + 1, page.get_text("text").strip()) for i, page in enumerate(pdf)]
        if not any(page.text for page in pages):
            pages = _ocr_pdf(path)
        return pages
    if suffix == ".docx":
        doc = DocxDocument(path)
        return [PageText(1, "\n".join(p.text for p in doc.paragraphs))]
    if suffix in {".txt", ".md"}:
        return [PageText(1, path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError("Supported types are PDF, DOCX, TXT, and Markdown.")


def _ocr_pdf(path: Path) -> list[PageText]:
    """OCR image-only PDFs when optional Tesseract dependencies are installed."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ValueError("This scanned PDF needs OCR. Install the optional `.[ocr]` dependencies and Tesseract.") from exc
    with fitz.open(path) as pdf:
        pages: list[PageText] = []
        for page_number, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pages.append(PageText(page_number, pytesseract.image_to_string(image).strip()))
    if not any(page.text for page in pages):
        raise ValueError("OCR completed but found no readable text.")
    return pages


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            breakpoint = max(clean.rfind(". ", start, end), clean.rfind("; ", start, end))
            if breakpoint > start + size // 2:
                end = breakpoint + 1
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = end - overlap
    return chunks


def ingest(path: Path, original_filename: str) -> dict:
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with connection() as conn:
        duplicate = conn.execute("SELECT id, filename FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
    if duplicate:
        raise FileExistsError(f"{duplicate['filename']} is already indexed (document {duplicate['id']}).")
    pages = extract_pages(path)
    document_id = str(uuid.uuid4())
    records: list[tuple[str, str, int, int, str, str, str]] = []
    ordinal = 0
    for page in pages:
        for piece in chunk_text(page.text):
            safe = sanitize_source_text(piece)
            if safe.strip():
                records.append((str(uuid.uuid4()), document_id, page.page, ordinal, piece, safe,
                                json.dumps(local_embed(safe))))
                ordinal += 1
    if not records:
        raise ValueError("No usable text was found in the document.")
    with connection() as conn:
        conn.execute("INSERT INTO documents (id, filename, page_count, created_at, content_hash) VALUES (?, ?, ?, ?, ?)",
                     (document_id, original_filename, len(pages), utcnow(), content_hash))
        conn.executemany("INSERT INTO chunks (id, document_id, page, ordinal, text, safe_text, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?)", records)
    return {"id": document_id, "filename": original_filename, "page_count": len(pages),
            "chunk_count": len(records), "created_at": utcnow()}
