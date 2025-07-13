
"""ingest.py — build a multimodal Chroma DB with Jina v4 embeddings + OCR

• Scans **./data/pdfs** for generic PDF documents (manuals, white papers, forms, …).
• Per page:
    ─ if the PDF already contains native text ⇒ emit text chunks (metadata.type="text")
    ─ otherwise (graphic/scan) ⇒
        • extract text via Tesseract OCR (metadata.type="ocr_text")
        • save a hi‑res JPEG (300 DPI) in **./images**
        • generate a thumbnail ≤ 11 000 base64 chars (metadata.type="image_thumbnail", metadata.hires_url)
• Uses conservative batching to respect Jina v4 payload limits.
"""
from __future__ import annotations

import base64
import os
import warnings
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_path
from pydantic import SecretStr
from PIL import Image
import pytesseract  # type: ignore

# ───────────────── bootstrap & config ─────────────────
load_dotenv()
warnings.simplefilter("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR    = PROJECT_ROOT / "db"
PDF_ROOT  = PROJECT_ROOT / "data" / "pdfs"   # generic PDF repository
HIRES_DIR = PROJECT_ROOT / "images"
HIRES_DIR.mkdir(parents=True, exist_ok=True)

MAX_B64_CHARS = 11_000   # ≈ 8 k tokens (Jina limit)
THUMB_W       = 800      # px thumbnail width
DPI           = 300      # images rendered at 300 DPI for OCR
TEXT_TH       = 40       # <40 characters ⇒ likely scanned page

# ───── Tesseract path (Windows override via env) ─────
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = os.getenv(
        "TESSERACT_EXE",
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    )

OCR_LANG = os.getenv("OCR_LANG", "eng+ita")

# ───────────────── helper: thumbnail under limit ─────────────────
def _b64_jpeg(img: Image.Image, quality: int) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def _thumbnail_b64(img: Image.Image) -> tuple[str, int, int] | None:
    """Return (b64, w, h) or None if thumbnail cannot fit under MAX_B64_CHARS."""
    scale = THUMB_W / img.width
    thumb = img.resize((THUMB_W, int(img.height * scale)), Image.Resampling.LANCZOS)
    for q in range(85, 35, -10):  # progressively lower quality
        b64 = _b64_jpeg(thumb, q)
        if len(b64) <= MAX_B64_CHARS:
            return b64, thumb.width, thumb.height
    return None

# ───────────────── main ingestion routine ─────────────────
def create_vector_database() -> None:
    pdf_paths = list(PDF_ROOT.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {PDF_ROOT}. Place your documents first.")

    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)

    for pdf_path in pdf_paths:
        print(f"\n— Processing {pdf_path.relative_to(PDF_ROOT.parent)}")
        # 1️⃣ Native text (PyPDFLoader)
        native_pages = PyPDFLoader(str(pdf_path)).load()
        # 2️⃣ Images for OCR + thumbnails
        images = convert_from_path(str(pdf_path), dpi=DPI, fmt="jpeg")

        for idx, page_doc in enumerate(native_pages):
            page_num = idx + 1
            page_img = images[idx]
            src_id   = f"{pdf_path.name}#p{page_num}"

            # — text extraction —
            text_content = (page_doc.page_content or "").strip()
            if len(text_content) >= TEXT_TH:
                for chunk in splitter.split_text(text_content):
                    docs.append(Document(page_content=chunk,
                                         metadata={"source": src_id, "type": "text"}))
            else:
                ocr_text = pytesseract.image_to_string(page_img, lang=OCR_LANG).strip()
                if ocr_text:
                    for chunk in splitter.split_text(ocr_text):
                        docs.append(Document(page_content=chunk,
                                             metadata={"source": src_id, "type": "ocr_text"}))

            # — store images —
            hires_name = f"{pdf_path.stem}_p{page_num}.jpg"
            hires_path = HIRES_DIR / hires_name
            if not hires_path.exists():
                page_img.save(hires_path, format="JPEG", quality=90)

            thumb = _thumbnail_b64(page_img)
            if thumb:
                b64, w, h = thumb
                docs.append(Document(
                    page_content=f"data:image/jpeg;base64,{b64}",
                    metadata={
                        "source": src_id,
                        "type": "image_thumbnail",
                        "w": w,
                        "h": h,
                        "hires_url": str(hires_path.relative_to(PROJECT_ROOT)),
                    },
                ))
            else:
                print(f"  ⚠ thumbnail too large for page {page_num}, skipped")


    print(f"Prepared {len(docs)} documents for embedding")

    # ───── embed & persist ─────
    jina_key = os.getenv("JINA_API_KEY")
    embeddings = JinaEmbeddings(
        session=requests.Session(),
        model_name="jina-embeddings-v4",
        jina_api_key=SecretStr(jina_key) if jina_key else None,
    )

    vectordb = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    BATCH, SUB = 8, 3
    total = ceil(len(docs) / BATCH)
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        try:
            print(f"adding batch {i//BATCH+1}/{total} …")
            vectordb.add_documents(batch)
        except Exception as e:
            print("  ⚠ batch failed →", e)
            for j in range(0, len(batch), SUB):
                try:
                    vectordb.add_documents(batch[j : j + SUB])
                except Exception as sub_e:
                    print("    ❌ sub‑batch", sub_e, "source", batch[j].metadata.get("source"))
    vectordb.persist()
    print("✅ Vector DB saved in", DB_DIR)


if __name__ == "__main__":
    create_vector_database()
