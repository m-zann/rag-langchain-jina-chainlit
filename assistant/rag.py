
"""RAG utilities.

Provides :func:`get_chain` which returns a RetrievalQA chain
ready to answer questions against the persisted vector DB.
"""
from __future__ import annotations

import os
import requests
from pathlib import Path
from typing import List

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "db"

class FilterImageRetriever(BaseRetriever, BaseModel):
    """Drop pure‑image docs before feeding context to the LLM."""
    base: BaseRetriever

    class Config:
        arbitrary_types_allowed = True

    # sync
    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):  # type: ignore[override]
        docs = self.base.get_relevant_documents(query)
        return [d for d in docs if not str(d.page_content).startswith("data:image")]

    # async
    async def _aget_relevant_documents(self, query, *, run_manager=None, **kwargs):  # type: ignore[override]
        docs = await self.base.aget_relevant_documents(query)
        return [d for d in docs if not str(d.page_content).startswith("data:image")]

def _load_model():
    return Ollama(
        model="mistral",
        verbose=False,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

def get_chain() -> RetrievalQA:
    """Return a configured RetrievalQA chain."""
    jina_key = os.getenv("JINA_API_KEY")
    embeddings = JinaEmbeddings(
        session=requests.Session(),
        model_name="jina-embeddings-v4",
        jina_api_key=SecretStr(jina_key) if jina_key else None,
    )

    vectordb = Chroma(
        collection_name="documents",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    base_retriever = vectordb.as_retriever(search_kwargs={"k": 20})
    retriever = FilterImageRetriever(base=base_retriever)

    return RetrievalQA.from_chain_type(
        _load_model(),
        retriever=retriever,
        return_source_documents=True,
    )
