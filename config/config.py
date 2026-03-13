"""
config/config.py
Central configuration for the NeoStats AI Career Advisor chatbot.
All API keys are loaded from environment variables — never hardcoded.
"""

import os

# ─────────────────────────────────────────────
# LLM Provider API Keys
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────
# Web Search API Keys
# ─────────────────────────────────────────────
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")

# ─────────────────────────────────────────────
# LLM Model Settings
# ─────────────────────────────────────────────
DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "groq")  # groq | openai | gemini

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

# ─────────────────────────────────────────────
# RAG / Embedding Settings
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "4"))

# FAISS index persistence path (relative to project root)
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")

# ─────────────────────────────────────────────
# Response Mode System Prompts
# ─────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an expert AI Career Advisor specializing in technology careers,
job market trends, resume guidance, interview preparation, and career transitions into AI/ML roles.
You have deep knowledge of the Indian and global tech job market.
Always be empathetic, professional, and actionable in your responses."""

CONCISE_SUFFIX = """
RESPONSE STYLE: Be concise. Respond in 3–5 sentences maximum.
Prioritize the single most important piece of advice. Avoid filler."""

DETAILED_SUFFIX = """
RESPONSE STYLE: Be comprehensive and structured.
Use clear headings, bullet points where helpful, and provide step-by-step guidance.
Include examples, resources, or next-action items where relevant."""

RAG_CONTEXT_TEMPLATE = """
Use the following retrieved knowledge base context to enhance your answer.
If the context is not directly relevant, rely on your own expertise.

CONTEXT:
{context}
"""

WEB_SEARCH_TEMPLATE = """
The following live web search results were retrieved to answer this question.
Synthesize and cite them where relevant.

WEB RESULTS:
{results}
"""

# ─────────────────────────────────────────────
# Application Settings
# ─────────────────────────────────────────────
APP_TITLE = "AI Career Advisor"
APP_ICON = "🎯"
APP_VERSION = "1.0.0"

MAX_CHAT_HISTORY = int(os.environ.get("MAX_CHAT_HISTORY", "20"))  # messages to keep

# Web search trigger keywords (if any appear in query, search is activated)
WEB_SEARCH_TRIGGERS = [
    "latest", "current", "recent", "2024", "2025", "2026",
    "today", "now", "trending", "news", "salary", "hiring",
    "job market", "demand", "openings"
]
