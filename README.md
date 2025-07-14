# Multimodal PDF Assistant 📄🤖

A **Retrieval‑Augmented Generation (RAG)** pipeline that turns any folder of PDFs—manuals, reports, invoices, even scanned paperwork—into a searchable knowledge base.  
Ask questions in a **Chainlit** chat and get answers with cited snippets *and* thumbnails of relevant diagrams.  
It has been created to handle home appliances manuals without having to constantly search for info in paper ones or in pdf files.  
Originally the choice of Jina has been made to leverage its multicontent embeddings, unfortunately local machine couldn't run it and API has some limitations, so we had to rely on OCR for non textual content.

---

## ✨  What You Get

| Stage | Tech | Purpose |
|-------|------|---------|
| **Ingest** | `pdf2image`, `pytesseract` | Extract native text, OCR scanned pages (300 DPI), save hi‑res JPEGs, embed tiny thumbnails |
| **Embeddings** | **Jina Embeddings v4** | Lightweight 8 k‑token context window, great for multimodal payloads |
| **Vector DB** | `Chroma` | Local on‑disk store (`./db`)—fast & private |
| **LLM** | `mistral` via **Ollama** | Runs fully offline, GPU‑accelerated |
| **UI** | **Chainlit** | Modern chat with expandable sources & inline images |

---

## 🛠  Prerequisites

| Component | Why | Install |
|-----------|-----|---------|
| **Python ≥ 3.9** | Runtime | <https://www.python.org/downloads/> |
| **Ollama** | Local LLM backend | <https://ollama.com/download>
| **`mistral` model** | Default 7 B model | `ollama pull mistral` *(after installing Ollama)* |
| **Tesseract‑OCR** | Accurate OCR for scanned PDFs | • Windows: "[Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)" → install, then add the *installation folder* to `%PATH%` or set `TESSERACT_EXE=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`.  
• macOS: `brew install tesseract`  
• Linux (Debian/Ubuntu): `sudo apt install tesseract-ocr` |
| **Jina Embeddings** | Text & image embeddings | *Two* options → see below |

### 🔑  Option A — Jina Cloud API  *(simplest)*
1. Sign up at <https://jina.ai> → *Dashboard › API Keys*.
2. Copy the key into your local `.env`:
   ```env
   JINA_API_KEY=sk-…
   ```
3. **Limitation:** Cloud API enforces an *~8 k‑token* total payload. To stay under the cap we create aggressively‑compressed thumbnails instead of full‑size images.

### 🖥️  Option B — Run Jina Locally *(no token limits)*
1. Follow the [official guide](https://github.com/jina-ai/embeddings) to spin up a local server (Docker‑compose or binary).
2. Remove/comment `JINA_API_KEY` from `.env`.
3. Because *local* doesn’t impose the 8 k cap, you can **set `THUMB_QUALITY=95`** in `.env` (or even disable thumbnails) and keep full‑resolution images; OCR still runs for scanned pages.

---

## 🚀  Installation & First Run

> The following assumes **Unix‑like shell**. For Windows CMD/Powershell just translate `source` → `venv\Scripts\activate` and use backslashes.

```bash
# 1️⃣  Clone & enter
$ git clone https://github.com/m-zann/rag-langchain-jina-chainlit.git
$ cd pdf‑assistant

# 2️⃣  Create & activate a virtualenv (recommended)
$ python -m venv .venv
$ source .venv/bin/activate

# 3️⃣  Install project in editable mode
$ pip install -e .

# 4️⃣  Copy env template and fill in the blanks
$ cp .env.example .env
#    -> add JINA_API_KEY (or not)
#    -> set TESSERACT_EXE on Windows if needed

# 5️⃣  Drop your PDFs
$ mkdir -p data/pdfs
$ mv ~/Downloads/*.pdf data/pdfs/

# 6️⃣  Build / update the vector DB (can be rerun any time)
$ python -m assistant.ingest  # takes a few minutes depending on pages

# 7️⃣  Chat!
$ chainlit run chainlit_app/main.py
```

**Pro tip:** `assistant.ingest` is idempotent; it only re‑processes new or modified PDFs.

---

## 💬  Usage Patterns

### Chainlit UI
* Type natural‑language questions like **“What does the error light mean on model XYZ?”**  
* Click on the *source_0*, *source_1*… pills to expand the exact PDF snippets, or on *img_0* to display a referenced diagram.

### CLI (headless)
```bash
$ python -m assistant.cli "How to clean the filter on ABC123?"
```
Use the `--ingest` flag to force a fresh ingest right before querying.

---

## 🖥️  GPU Acceleration (NVIDIA)

Ollama should detect CUDA automatically, but on Windows it might keep running on CPU, making the experience very slow:

1. **Force the GPU** in *NVIDIA Control Panel → Manage 3D Settings → Program Settings* → select `ollama.exe` → set *High‑performance NVIDIA processor*.
2. Override layers at runtime:
   ```powershell
   ollama run mistral --num-gpu-layers 35  # tweak the number for your VRAM
   ```

---

## 🧪  Running Tests

Tests are still to be completed, to run the current ones use the following command.

```bash
pytest -q
```

Tests spin up an in‑memory Chroma instance and use small fixture PDFs so they finish in seconds.

---

## 📜  License

MIT © 2024 Your Name or Organisation

