
# Multimodal PDF Assistant 📄🤖

A **RAG (Retrieval‑Augmented Generation)** assistant that indexes any folder of PDFs
— manuals, white papers, invoices, scanned forms — and lets you query them via
CLI or a sleek **Chainlit** UI.  
The pipeline is **multimodal**: native text is embedded directly, while scanned
pages are OCR‑ed with **Tesseract** and a thumbnail is stored so the LLM can
reference diagrams when needed.

---

## Features

| Stage | Tech | What happens |
|-------|------|--------------|
| Ingestion | `pdf2image`, `pytesseract` | • Extract native text<br>• OCR scanned pages (300 DPI)<br>• Save hi‑res JPEGs<br>• Add tiny base64 thumbnails |
| Embeddings | `Jina v4` | Optimised for small context windows (8 k tokens) |
| Vector DB | `Chroma` | Local, persisted in **./db** |
| LLM | `mistral` via **Ollama** | Works offline, GPU‑accelerated |
| UI | **Chainlit** | Chat UX with expandable sources and inline images |

---

## Quick start

```bash
git clone https://github.com/your‑org/pdf‑assistant.git
cd pdf‑assistant

# 1️⃣  Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2️⃣  Configure
cp .env.example .env
#   – add your JINA_API_KEY
#   – (Windows) set TESSERACT_EXE
#   – adjust OCR_LANG if needed

# 3️⃣  Drop PDFs
mkdir -p data/pdfs
cp ~/Downloads/*.pdf data/pdfs/

# 4️⃣  Build the DB
python -m assistant.ingest

# 5️⃣  Fire up the chat
chainlit run chainlit_app/main.py
```

### CLI usage

```bash
python -m assistant.cli --ingest "How do I reset the filter on model ABC123?"
```

*Add `--ingest` to rebuild before querying.*

### Tests

```bash
pytest -q
```

---

## Project Layout

```
project/
├─ assistant/          # ingestion + RAG code
│  ├─ ingest.py
│  ├─ rag.py
│  └─ cli.py
├─ chainlit_app/
│  └─ main.py          # Chainlit UI
├─ data/pdfs/          # put your PDFs here (ignored by Git)
├─ images/             # hi‑res JPEGs extracted from PDFs
├─ db/                 # Chroma DB
├─ tests/              # minimal pytest suite
├─ .env.example
└─ pyproject.toml
```

## GPU tips (NVIDIA)

1. **Ollama** automatically detects CUDA. On Windows, ensure *nvidia‑smi* is visible
   in `%PATH%` and run:

   ```powershell
   ollama run mistral --num-gpu-layers 35
   ```

2. If the GUI still spins up a CPU instance, open *NVIDIA Control Panel → Manage
   3D Settings → Program Settings* and force *ollama.exe* to the high‑perf GPU.

---

## License

MIT
