"""
utils/document_loader.py
Loads and splits documents from various sources:
  - Local text/PDF/markdown files
  - Uploaded Streamlit files (BytesIO)
  - Plain text strings

Returns a list of LangChain Document objects ready for embedding.
"""

import os
import tempfile
from typing import List, Union
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Returns the configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_text_file(filepath: str) -> List[Document]:
    """Load a plain .txt or .md file from disk."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        filename = os.path.basename(filepath)
        splitter = _get_splitter()
        chunks = splitter.split_text(text)

        return [
            Document(page_content=chunk, metadata={"source": filename, "type": "text"})
            for chunk in chunks
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to load text file '{filepath}': {e}")


def load_pdf_file(filepath: str) -> List[Document]:
    """Load a PDF file from disk using PyPDF2 or pdfplumber."""
    try:
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append((i + 1, text))
        except ImportError:
            # Fallback to PyPDF2
            import PyPDF2

            pages = []
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append((i + 1, text))

        filename = os.path.basename(filepath)
        splitter = _get_splitter()
        docs = []

        for page_num, text in pages:
            if text.strip():
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "type": "pdf",
                            },
                        )
                    )

        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{filepath}': {e}")


def load_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a document from raw bytes (e.g., Streamlit uploaded file).
    Detects file type by extension.
    """
    try:
        ext = os.path.splitext(filename)[-1].lower()

        # Write to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if ext == ".pdf":
                docs = load_pdf_file(tmp_path)
            else:
                docs = load_text_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Update metadata with original filename
        for doc in docs:
            doc.metadata["source"] = filename

        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load uploaded file '{filename}': {e}")


def load_from_string(text: str, source: str = "inline") -> List[Document]:
    """
    Create documents from a plain text string.
    Useful for injecting static knowledge.
    """
    try:
        splitter = _get_splitter()
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={"source": source, "type": "string"},
            )
            for chunk in chunks
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to load text string: {e}")


def load_directory(directory: str, extensions: List[str] = None) -> List[Document]:
    """
    Recursively load all supported documents from a directory.

    Args:
        directory: path to the folder
        extensions: list of extensions to include, e.g. ['.txt', '.pdf']
                    defaults to ['.txt', '.md', '.pdf']

    Returns:
        List of Document objects
    """
    if extensions is None:
        extensions = [".txt", ".md", ".pdf"]

    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: '{directory}'")

    all_docs = []
    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[-1].lower()
            if ext in extensions:
                fpath = os.path.join(root, fname)
                try:
                    if ext == ".pdf":
                        docs = load_pdf_file(fpath)
                    else:
                        docs = load_text_file(fpath)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"[DocumentLoader] Warning: skipped '{fpath}' — {e}")

    return all_docs
