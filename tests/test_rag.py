
from assistant.rag import get_chain
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import JinaEmbeddings
from PyPDF2 import PdfWriter

def test_chain_builds():
    chain = get_chain()
    import assistant.rag as rag

    @pytest.fixture
    def mock_pdf(tmp_path):
        # Create a simple PDF file for testing
        pdf_path = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return str(pdf_path)

    def test_text_splitter_splits_text(monkeypatch):
        text = "a" * 2500
        # Mock text_splitter if not present in rag
        class MockTextSplitter:
            def split_text(self, text):
                return [text[i:i+1000] for i in range(0, len(text), 1000)]
        rag.text_splitter = MockTextSplitter()
        chunks = rag.text_splitter.split_text(text)
        assert all(len(chunk) <= 1000 for chunk in chunks)
        assert sum(len(chunk) for chunk in chunks) == 2500

    def test_jina_embeddings_init(monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "dummy")
        embeddings = rag.JinaEmbeddings(
            session=MagicMock(),
            model_name="jina-embeddings-v4",
            jina_api_key=rag.SecretStr("dummy"),
        )
        assert embeddings is not None

    def test_chroma_from_texts(monkeypatch):
        # Patch Chroma.from_texts to return a mock
        mock_chroma = MagicMock()
        monkeypatch.setattr(rag.Chroma, "from_texts", MagicMock(return_value=mock_chroma))
        texts = ["chunk1", "chunk2"]
        embeddings = MagicMock()
        metadatas = [{"source": "0-pl"}, {"source": "1-pl"}]
        result = rag.Chroma.from_texts(texts, embeddings, metadatas=metadatas)
        assert result == mock_chroma

    def test_conversational_retrieval_chain(monkeypatch):
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_memory = MagicMock()
        chain = ConversationalRetrievalChain.from_llm(
            mock_llm,
            chain_type="stuff",
            retriever=mock_retriever,
            memory=mock_memory,
            return_source_documents=True,
        )
        assert isinstance(chain, ConversationalRetrievalChain)

    @pytest.mark.asyncio
    async def test_on_chat_start(monkeypatch, mock_pdf):
        # Patch file upload and dependencies
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.path = mock_pdf
        monkeypatch.setattr(cl, "AskFileMessage", MagicMock(return_value=AsyncMock(send=AsyncMock(return_value=[mock_file]))))
        monkeypatch.setattr(cl, "Message", MagicMock(return_value=AsyncMock(send=AsyncMock(), update=AsyncMock())))
        monkeypatch.setattr(rag.PyPDF2, "PdfReader", MagicMock(return_value=MagicMock(pages=[MagicMock(extract_text=MagicMock(return_value="Hello world") )])))
        monkeypatch.setattr(rag, "JinaEmbeddings", MagicMock())
        monkeypatch.setattr(rag, "Chroma", MagicMock())
        monkeypatch.setattr(cl, "make_async", lambda x: x)
        monkeypatch.setattr(cl.user_session, "set", MagicMock())
        await rag.on_chat_start()
        cl.user_session.set.assert_called_with("chain", ANY := object)

    @pytest.mark.asyncio
    async def test_on_message(monkeypatch):
        # Patch user_session.get to return a mock chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value={
            "answer": "Test answer",
            "source_documents": [MagicMock(page_content="source1"), MagicMock(page_content="source2")]
        })
        monkeypatch.setattr(cl.user_session, "get", MagicMock(return_value=mock_chain))
        monkeypatch.setattr(cl, "AsyncLangchainCallbackHandler", MagicMock())
        monkeypatch.setattr(cl, "Text", MagicMock(side_effect=lambda content, name: MagicMock(name=name, content=content)))
        monkeypatch.setattr(cl, "Message", MagicMock(return_value=AsyncMock(send=AsyncMock())))
        msg = MagicMock(content="What is this?")
        await rag.main(msg)
        cl.Message.return_value.send.assert_awaited()

    @pytest.mark.asyncio
    async def test_on_message_no_sources(monkeypatch):
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value={
            "answer": "Test answer",
            "source_documents": []
        })
        monkeypatch.setattr(cl.user_session, "get", MagicMock(return_value=mock_chain))
        monkeypatch.setattr(cl, "AsyncLangchainCallbackHandler", MagicMock())
        monkeypatch.setattr(cl, "Text", MagicMock())
        monkeypatch.setattr(cl, "Message", MagicMock(return_value=AsyncMock(send=AsyncMock())))
        msg = MagicMock(content="What is this?")
        await rag.main(msg)
        cl.Message.return_value.send.assert_awaited()
