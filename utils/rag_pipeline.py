"""
utils/rag_pipeline.py
FAISS-backed RAG pipeline.

Responsibilities:
  - Build a FAISS vector store from documents
  - Persist / load the index to disk
  - Retrieve top-K relevant chunks for a query
  - Format retrieved chunks into a context string for the LLM
"""

import os
import sys
from typing import List, Optional

from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import FAISS_INDEX_PATH, TOP_K_RETRIEVAL


class RAGPipeline:
    """
    Manages a FAISS vector store for document retrieval.

    Usage:
        pipeline = RAGPipeline(embeddings)
        pipeline.build_from_documents(docs)
        context = pipeline.retrieve_context("How do I switch to ML?")
    """

    def __init__(self, embeddings, index_path: str = None):
        """
        Args:
            embeddings: LangChain Embeddings instance
            index_path: path to persist/load FAISS index
        """
        self.embeddings = embeddings
        self.index_path = index_path or FAISS_INDEX_PATH
        self.vectorstore = None

    def build_from_documents(self, documents: List[Document]) -> None:
        """
        Build FAISS index from a list of LangChain Documents.
        Automatically persists the index to disk.

        Args:
            documents: list of chunked Document objects
        """
        try:
            from langchain_community.vectorstores import FAISS

            if not documents:
                raise ValueError("No documents provided to build the vector store.")

            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self._save()
            print(f"[RAG] Built index with {len(documents)} chunks.")
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add more documents to an existing vector store (incremental update).
        """
        try:
            if self.vectorstore is None:
                self.build_from_documents(documents)
            else:
                self.vectorstore.add_documents(documents)
                self._save()
                print(f"[RAG] Added {len(documents)} chunks to existing index.")
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {e}")

    def load(self) -> bool:
        """
        Load a persisted FAISS index from disk.

        Returns:
            True if loaded successfully, False if not found.
        """
        try:
            from langchain_community.vectorstores import FAISS

            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[RAG] Loaded index from '{self.index_path}'.")
                return True
            return False
        except Exception as e:
            print(f"[RAG] Warning: Could not load index — {e}")
            return False

    def _save(self) -> None:
        """Persist the current vector store to disk."""
        try:
            if self.vectorstore is not None:
                os.makedirs(self.index_path, exist_ok=True)
                self.vectorstore.save_local(self.index_path)
        except Exception as e:
            print(f"[RAG] Warning: Could not save index — {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: user's question
            top_k: number of chunks to retrieve

        Returns:
            List of Document objects sorted by relevance
        """
        if self.vectorstore is None:
            return []

        try:
            k = top_k or TOP_K_RETRIEVAL
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return []

    def retrieve_context(self, query: str, top_k: int = None) -> str:
        """
        Retrieve and format top-K chunks into a single context string.

        Args:
            query: user's question
            top_k: number of chunks to retrieve

        Returns:
            Formatted context string (empty string if nothing found)
        """
        docs = self.retrieve(query, top_k=top_k)

        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def is_ready(self) -> bool:
        """Returns True if a vector store is loaded/built and ready."""
        return self.vectorstore is not None

    def get_document_count(self) -> int:
        """Returns approximate number of vectors in the store."""
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore.index.ntotal
        except Exception:
            return 0
