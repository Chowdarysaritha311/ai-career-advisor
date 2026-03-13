"""
models/embeddings.py
Embedding model factory supporting sentence-transformers and OpenAI embeddings.
All embedding logic is centralized here.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_sentence_transformer_embeddings(model_name: str = None):
    """
    Returns a HuggingFace sentence-transformer embedding model.
    Works fully offline — no API key needed.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = model_name or EMBEDDING_MODEL
        embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except ImportError:
        # Fallback to older langchain community import
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model = model_name or EMBEDDING_MODEL
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace embeddings: {e}")


def get_openai_embeddings(api_key: str = None, model: str = "text-embedding-3-small"):
    """
    Returns an OpenAI embedding model.

    Args:
        api_key: OpenAI API key (falls back to config)
        model: OpenAI embedding model name

    Returns:
        OpenAIEmbeddings instance
    """
    try:
        from langchain_openai import OpenAIEmbeddings

        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")

        return OpenAIEmbeddings(api_key=key, model=model)
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenAI embeddings: {e}")


def get_embeddings(provider: str = "huggingface", **kwargs):
    """
    Unified embedding factory.

    Args:
        provider: 'huggingface' | 'openai'
        **kwargs: passed to the underlying embedding constructor

    Returns:
        LangChain Embeddings instance
    """
    provider = provider.lower()

    if provider == "huggingface":
        return get_sentence_transformer_embeddings(
            model_name=kwargs.get("model_name")
        )
    elif provider == "openai":
        return get_openai_embeddings(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "text-embedding-3-small"),
        )
    else:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. Choose: 'huggingface' | 'openai'"
        )
