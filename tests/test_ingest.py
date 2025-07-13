
from assistant.ingest import create_vector_database
from pathlib import Path

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
        pass
