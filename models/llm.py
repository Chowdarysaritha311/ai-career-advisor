"""
models/llm.py
LLM factory supporting Groq, OpenAI, and Google Gemini.
Returns a LangChain-compatible chat model instance.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
    GROQ_MODEL, OPENAI_MODEL, GEMINI_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, DEFAULT_PROVIDER,
)


def get_chatgroq_model(api_key: str = None, model: str = None):
    """
    Returns a ChatGroq LangChain model.
    Falls back to config values if not provided.
    """
    try:
        from langchain_groq import ChatGroq

        key = api_key or GROQ_API_KEY
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")

        return ChatGroq(
            api_key=key,
            model=model or GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {e}")


def get_openai_model(api_key: str = None, model: str = None):
    """
    Returns a ChatOpenAI LangChain model.
    Falls back to config values if not provided.
    """
    try:
        from langchain_openai import ChatOpenAI

        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")

        return ChatOpenAI(
            api_key=key,
            model=model or OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {e}")


def get_gemini_model(api_key: str = None, model: str = None):
    """
    Returns a ChatGoogleGenerativeAI LangChain model.
    Falls back to config values if not provided.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError("GOOGLE_API_KEY is not set.")

        return ChatGoogleGenerativeAI(
            google_api_key=key,
            model=model or GEMINI_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {e}")


def get_llm(
    provider: str = None,
    api_key: str = None,
    model: str = None,
):
    """
    Unified LLM factory. Returns the appropriate chat model
    based on provider name.

    Args:
        provider: 'groq' | 'openai' | 'gemini'
        api_key: override API key
        model: override model name

    Returns:
        LangChain BaseChatModel instance
    """
    provider = (provider or DEFAULT_PROVIDER).lower()

    dispatch = {
        "groq": get_chatgroq_model,
        "openai": get_openai_model,
        "gemini": get_gemini_model,
    }

    if provider not in dispatch:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(dispatch.keys())}"
        )

    return dispatch[provider](api_key=api_key, model=model)
