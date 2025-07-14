
from assistant.ingest import create_vector_database
from pathlib import Path
import io
import os
import types
import base64
import tempfile
import shutil
import pytest
from PIL import Image

def test_ingest_runs(tmp_path, monkeypatch):
    # Point the PDF_ROOT to a tmp directory with dummy PDF
    sample_pdf = tmp_path / "dummy.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    monkeypatch.setattr("assistant.ingest.PDF_ROOT", tmp_path)
    monkeypatch.setattr("assistant.ingest.DB_DIR", tmp_path / "db")
    try:
        create_vector_database()
    except RuntimeError:
        # expected because dummy PDF has no pages
        import assistant.ingest as ingest

        @pytest.fixture
        def dummy_pdf_path(tmp_path):
            pdf = tmp_path / "dummy.pdf"
            pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
            return pdf

        def test_b64_jpeg_and_thumbnail_b64(tmp_path):
            img = Image.new("RGB", (1000, 2000), color="white")
            b64 = ingest._b64_jpeg(img, 80)
            assert isinstance(b64, str)
            thumb = ingest._thumbnail_b64(img)
            assert thumb is not None
            b64, w, h = thumb
            assert isinstance(b64, str)
            assert w == ingest.THUMB_W
            assert h > 0

        def test_thumbnail_b64_too_large(monkeypatch):
            img = Image.new("RGB", (10000, 20000), color="white")
            # Patch _b64_jpeg to always return a string that's too long
            monkeypatch.setattr(ingest, "_b64_jpeg", lambda img, q: "A" * (ingest.MAX_B64_CHARS + 1))
            assert ingest._thumbnail_b64(img) is None

        def test_create_vector_database_no_pdfs(tmp_path, monkeypatch):
            monkeypatch.setattr(ingest, "PDF_ROOT", tmp_path)
            with pytest.raises(RuntimeError):
                ingest.create_vector_database()

        def test_create_vector_database_native_and_ocr(monkeypatch, tmp_path):
            # Setup dummy PDF path
            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
            monkeypatch.setattr(ingest, "PDF_ROOT", tmp_path)
            monkeypatch.setattr(ingest, "DB_DIR", tmp_path / "db")
            monkeypatch.setattr(ingest, "HIRES_DIR", tmp_path / "images")
            ingest.HIRES_DIR.mkdir(exist_ok=True)
            # Patch PyPDFLoader to return fake pages
            class DummyDoc:
                def __init__(self, content):
                    self.page_content = content
            monkeypatch.setattr(ingest, "PyPDFLoader", lambda path: types.SimpleNamespace(load=lambda: [
                DummyDoc("This is a native text page with enough content to pass threshold."),
                DummyDoc(""),
            ]))
            # Patch convert_from_path to return PIL images
            img = Image.new("RGB", (1000, 2000), color="white")
            monkeypatch.setattr(ingest, "convert_from_path", lambda *a, **k: [img, img])
            # Patch pytesseract.image_to_string to return OCR text
            monkeypatch.setattr(ingest.pytesseract, "image_to_string", lambda img, lang=None: "OCR text content")
            # Patch JinaEmbeddings and Chroma
            class DummyEmbeddings:
                def __init__(self, **kwargs): pass
            class DummyChroma:
                def __init__(self, **kwargs): self.docs = []
                def add_documents(self, docs): self.docs.extend(docs)
                def persist(self): self.persisted = True
            monkeypatch.setattr(ingest, "JinaEmbeddings", DummyEmbeddings)
            monkeypatch.setattr(ingest, "Chroma", DummyChroma)
            # Patch print to suppress output
            monkeypatch.setattr("builtins.print", lambda *a, **k: None)
            ingest.create_vector_database()

        def test_create_vector_database_batch_fail(monkeypatch, tmp_path):
            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
            monkeypatch.setattr(ingest, "PDF_ROOT", tmp_path)
            monkeypatch.setattr(ingest, "DB_DIR", tmp_path / "db")
            monkeypatch.setattr(ingest, "HIRES_DIR", tmp_path / "images")
            ingest.HIRES_DIR.mkdir(exist_ok=True)
            class DummyDoc:
                def __init__(self, content): self.page_content = content
            monkeypatch.setattr(ingest, "PyPDFLoader", lambda path: types.SimpleNamespace(load=lambda: [DummyDoc("A"*50)]*10))
            img = Image.new("RGB", (1000, 2000), color="white")
            monkeypatch.setattr(ingest, "convert_from_path", lambda *a, **k: [img]*10)
            monkeypatch.setattr(ingest.pytesseract, "image_to_string", lambda img, lang=None: "OCR text content")
            monkeypatch.setattr(ingest, "JinaEmbeddings", lambda **kwargs: None)
            # Chroma that fails on add_documents for batch, but works for sub-batch
            class DummyChroma:
                def __init__(self, **kwargs): self.calls = []
                def add_documents(self, docs):
                    if len(docs) > 3: raise Exception("batch too large")
                    self.calls.append(len(docs))
                def persist(self): pass
            monkeypatch.setattr(ingest, "Chroma", DummyChroma)
            monkeypatch.setattr("builtins.print", lambda *a, **k: None)
            ingest.create_vector_database()

        def test_main_guard(monkeypatch):
            monkeypatch.setattr(ingest, "create_vector_database", lambda: (_ for _ in ()).throw(SystemExit))
            with pytest.raises(SystemExit):
                exec(
                    "if __name__ == '__main__':\n    import assistant.ingest as i; i.create_vector_database()",
                    {"__name__": "__main__"}
                )
