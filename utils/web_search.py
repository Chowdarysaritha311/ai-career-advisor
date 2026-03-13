"""
utils/web_search.py
Web search utility supporting Tavily (primary) and DuckDuckGo (free fallback).

Search is triggered automatically when:
  - User query contains trigger keywords (e.g. "latest", "current", "salary")
  - RAG retrieval returns empty or low-confidence results
  - User explicitly requests web search

Results are summarized and formatted as context for the LLM.
"""

import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import TAVILY_API_KEY, SERPAPI_KEY, WEB_SEARCH_TRIGGERS


def should_trigger_search(query: str) -> bool:
    """
    Decide if web search should be automatically triggered for this query.

    Args:
        query: user's input text

    Returns:
        True if any trigger keyword is found (case-insensitive)
    """
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in WEB_SEARCH_TRIGGERS)


def search_tavily(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search using Tavily API (recommended — built for RAG/LLM use cases).

    Args:
        query: search query string
        max_results: maximum number of results to return

    Returns:
        List of dicts: [{"title": ..., "url": ..., "content": ...}]
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set.")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )

        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
            )
        return results

    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}")


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search using DuckDuckGo (free, no API key required).
    Falls back to this when Tavily is unavailable.

    Args:
        query: search query string
        max_results: maximum number of results to return

    Returns:
        List of dicts: [{"title": ..., "url": ..., "content": ...}]
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                    }
                )
        return results

    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search failed: {e}")


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Unified web search — tries Tavily first, falls back to DuckDuckGo.

    Args:
        query: search query string
        max_results: max results to return

    Returns:
        List of result dicts with 'title', 'url', 'content'
    """
    if TAVILY_API_KEY:
        try:
            return search_tavily(query, max_results=max_results)
        except Exception as e:
            print(f"[WebSearch] Tavily failed, falling back to DuckDuckGo: {e}")

    # Fallback
    return search_duckduckgo(query, max_results=max_results)


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results into a clean context string for the LLM.

    Args:
        results: list of result dicts from web_search()

    Returns:
        Formatted multi-line string
    """
    if not results:
        return "No web search results found."

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "").strip()

        parts.append(
            f"[Result {i}] {title}\nURL: {url}\n{content}"
        )

    return "\n\n---\n\n".join(parts)


def get_search_context(query: str, max_results: int = 5) -> tuple[str, bool]:
    """
    Main entry point for web search context generation.

    Args:
        query: user query
        max_results: number of results

    Returns:
        (formatted_context_string, search_was_performed)
    """
    try:
        results = web_search(query, max_results=max_results)
        return format_search_results(results), True
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return "", False
