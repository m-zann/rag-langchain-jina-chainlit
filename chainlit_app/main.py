# chainlit_app/main.py
from __future__ import annotations

import asyncio
import logging
import os
from typing import List

import chainlit as cl
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.runnables import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import BaseModel, SecretStr
from typing import Any, List

# --------------------------------------------------------------------------- #
# Config & logging                                                            #
# --------------------------------------------------------------------------- #
load_dotenv()

logging.getLogger("chainlit.i18n").setLevel(logging.ERROR)
logging.getLogger("chainlit").setLevel(logging.WARNING)

ABS_PATH   = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH  = os.path.dirname(ABS_PATH)
DB_DIR     = os.path.join(ROOT_PATH, "db")
COLLECTION = "documents"          # deve combaciare con ingest.py
MAX_QUERY  = 1_000                # limita input lunghissimi
MAX_SNIPS  = 5                    # max snippet testuali da mostrare

# --------------------------------------------------------------------------- #
# Prompt RAG (lo stesso che usavamo prima)                                    #
# --------------------------------------------------------------------------- #
rag_prompt = hub.pull("rlm/rag-prompt-mistral")

# --------------------------------------------------------------------------- #
# LLM helper                                                                  #
# --------------------------------------------------------------------------- #
def load_model() -> Ollama:
    return Ollama(
        model="mistral",
        verbose=False,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

# --------------------------------------------------------------------------- #
# Custom retriever che filtra i doc-immagine in fase di ragionamento          #
# --------------------------------------------------------------------------- #

class FilterImageRetriever(BaseRetriever):
    """Filtra i docs che sono solo miniature (data:image…)."""

    base: BaseRetriever  # retriever “vero” su cui appoggiarsi

    # Implement the required abstract method
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> List[Document]:
        if run_manager is not None:
            docs = self.base._get_relevant_documents(query, run_manager=run_manager, **kwargs)
        else:
            docs = self.base._get_relevant_documents(query, **kwargs)
        return [d for d in docs if not str(d.page_content).startswith("data:image")]

    # ───────── sync ─────────
    def invoke(
        self,
        input: str,                                 # 🟢 stesso nome/ordine del padre
        config: RunnableConfig | None = None,       # 🟢 stesso tipo
        **kwargs: Any,
    ) -> List[Document]:
        docs = self.base.invoke(input, config=config, **kwargs)
        return [d for d in docs if not str(d.page_content).startswith("data:image")]

    # ───────── async ─────────
    async def ainvoke(
        self,
        input: str,                                 # idem
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs = await self.base.ainvoke(input, config=config, **kwargs)
        return [d for d in docs if not str(d.page_content).startswith("data:image")]

# --------------------------------------------------------------------------- #
# Build RAG chain                                                             #
# --------------------------------------------------------------------------- #
def build_chain() -> RetrievalQA:
    jina_key = os.getenv("JINA_API_KEY")
    embeddings = JinaEmbeddings(
        session=requests.Session(),
        model_name="jina-embeddings-v4",
        jina_api_key=SecretStr(jina_key) if jina_key else None,
    )

    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    base_retriever = vectordb.as_retriever(search_kwargs={"k": 20})
    retriever      = FilterImageRetriever(base=base_retriever)

    return RetrievalQA.from_chain_type(
        llm=load_model(),
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
    )

# --------------------------------------------------------------------------- #
# Chainlit hooks                                                              #
# --------------------------------------------------------------------------- #
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chain", build_chain())
    await cl.Message(
        content=(
            "🛠️ **Pronto!** Chiedimi qualsiasi cosa sui PDF che hai indicizzato. "
            "Mostrerò le pagine (testo o miniature) più pertinenti come fonti."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    chain: RetrievalQA = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    query = message.content.strip()[:MAX_QUERY]

    try:
        res = await asyncio.wait_for(chain.ainvoke(query, callbacks=[cb]), timeout=300)
    except Exception as e:
        await cl.Message(content=f"❌ **Error**\n\n```{type(e).__name__}: {e}```").send()
        return

    answer  : str              = res["result"]
    sources : List             = res.get("source_documents", [])

    elements: List[cl.Element] = []
    names   : List[str]        = []

    # -- testi prima --
    for idx, doc in enumerate(sources):
        if len(elements) >= MAX_SNIPS:
            break
        if str(doc.page_content).startswith("data:image"):
            continue
        name = f"src_{idx}"
        snippet = doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else "")
        elements.append(cl.Text(content=snippet, name=name))
        names.append(name)

    # -- se nessun testo, mostra la prima immagine/thm --
    if not elements:
        for idx, doc in enumerate(sources):
            if str(doc.page_content).startswith("data:image"):
                name = f"img_{idx}"
                elements.append(cl.Image(content=doc.page_content, name=name))
                names.append(name)
                break

    if names:
        answer += "\n\n**Fonti**: " + ", ".join(names)

    await cl.Message(content=answer, elements=elements).send()
