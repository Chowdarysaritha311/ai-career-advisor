"""
app.py
NeoStats AI Career Advisor — Main Streamlit Application

Features:
  ✅ Multi-provider LLM (Groq / OpenAI / Gemini)
  ✅ RAG pipeline (FAISS + sentence-transformers)
  ✅ Live web search (Tavily / DuckDuckGo fallback)
  ✅ Response mode toggle (Concise / Detailed)
  ✅ Document upload for RAG
  ✅ Full chat history
  ✅ Production-grade error handling
"""

import os
import sys
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import (
    APP_TITLE, APP_ICON, APP_VERSION,
    BASE_SYSTEM_PROMPT, CONCISE_SUFFIX, DETAILED_SUFFIX,
    RAG_CONTEXT_TEMPLATE, WEB_SEARCH_TEMPLATE,
    MAX_CHAT_HISTORY, GROQ_API_KEY, OPENAI_API_KEY,
    GOOGLE_API_KEY, TAVILY_API_KEY,
)
from models.llm import get_llm
from models.embeddings import get_embeddings
from utils.document_loader import load_from_bytes, load_directory
from utils.rag_pipeline import RAGPipeline
from utils.web_search import should_trigger_search, get_search_context


# ─────────────────────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_TITLE} v{APP_VERSION}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "messages": [],
        "rag_pipeline": None,
        "rag_ready": False,
        "rag_doc_count": 0,
        "provider": "groq",
        "model_override": "",
        "api_key_override": "",
        "response_mode": "Detailed",
        "web_search_enabled": True,
        "last_used_search": False,
        "last_used_rag": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─────────────────────────────────────────────────────────────
# Helper: Build system prompt based on mode + context
# ─────────────────────────────────────────────────────────────
def build_system_prompt(
    mode: str,
    rag_context: str = "",
    web_context: str = "",
) -> str:
    prompt = BASE_SYSTEM_PROMPT

    if mode == "Concise":
        prompt += CONCISE_SUFFIX
    else:
        prompt += DETAILED_SUFFIX

    if rag_context:
        prompt += "\n\n" + RAG_CONTEXT_TEMPLATE.format(context=rag_context)

    if web_context:
        prompt += "\n\n" + WEB_SEARCH_TEMPLATE.format(results=web_context)

    return prompt


# ─────────────────────────────────────────────────────────────
# Helper: Get chat response from LLM
# ─────────────────────────────────────────────────────────────
def get_chat_response(
    chat_model,
    messages: list,
    system_prompt: str,
) -> str:
    try:
        formatted = [SystemMessage(content=system_prompt)]

        # Only keep last MAX_CHAT_HISTORY messages to avoid token overflow
        recent = messages[-MAX_CHAT_HISTORY:]
        for msg in recent:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted)
        return response.content

    except Exception as e:
        return f"⚠️ Error getting response: {str(e)}"


# ─────────────────────────────────────────────────────────────
# Helper: Initialise / reload RAG pipeline
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    """Cache the embedding model — expensive to reload."""
    try:
        return get_embeddings(provider="huggingface")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


def initialise_rag(docs=None):
    """Build or reload the RAG pipeline from documents."""
    embeddings = load_embeddings_model()
    if embeddings is None:
        return

    pipeline = RAGPipeline(embeddings=embeddings)

    # Try loading existing index
    loaded = pipeline.load()

    if docs:
        pipeline.add_documents(docs)
        st.session_state.rag_ready = True
    elif loaded:
        st.session_state.rag_ready = True
    else:
        # Load from default data directory if available
        try:
            default_docs = load_directory("data/sample_docs")
            if default_docs:
                pipeline.build_from_documents(default_docs)
                st.session_state.rag_ready = True
        except Exception:
            pass

    st.session_state.rag_pipeline = pipeline
    st.session_state.rag_doc_count = pipeline.get_document_count()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption(f"v{APP_VERSION} — Powered by LangChain")
        st.divider()

        # ── LLM Configuration ──────────────────────────────
        st.subheader("🤖 LLM Settings")

        provider = st.selectbox(
            "Provider",
            ["groq", "openai", "gemini"],
            index=["groq", "openai", "gemini"].index(st.session_state.provider),
        )
        st.session_state.provider = provider

        api_key_input = st.text_input(
            "API Key (leave blank to use env var)",
            value=st.session_state.api_key_override,
            type="password",
            help="Overrides the environment variable for this session.",
        )
        st.session_state.api_key_override = api_key_input

        model_input = st.text_input(
            "Model (leave blank for default)",
            value=st.session_state.model_override,
            placeholder="e.g. llama-3.1-70b-versatile",
        )
        st.session_state.model_override = model_input

        st.divider()

        # ── Response Mode ──────────────────────────────────
        st.subheader("⚡ Response Mode")
        mode = st.radio(
            "",
            ["Concise", "Detailed"],
            index=0 if st.session_state.response_mode == "Concise" else 1,
            help="Concise: 3–5 sentences. Detailed: full structured answer.",
        )
        st.session_state.response_mode = mode

        if mode == "Concise":
            st.caption("📌 Short, direct answers — 3–5 sentences max.")
        else:
            st.caption("📖 Comprehensive structured responses with examples.")

        st.divider()

        # ── Web Search ─────────────────────────────────────
        st.subheader("🌐 Web Search")
        web_enabled = st.toggle(
            "Enable Live Web Search",
            value=st.session_state.web_search_enabled,
        )
        st.session_state.web_search_enabled = web_enabled

        if web_enabled:
            if TAVILY_API_KEY:
                st.caption("✅ Tavily API connected")
            else:
                st.caption("⚠️ No Tavily key — using DuckDuckGo fallback")

        st.divider()

        # ── Document Upload for RAG ────────────────────────
        st.subheader("📚 Knowledge Base (RAG)")

        if st.session_state.rag_ready:
            st.success(f"✅ Index ready — {st.session_state.rag_doc_count} chunks")
        else:
            st.info("No index loaded yet.")

        uploaded_files = st.file_uploader(
            "Upload documents (.txt, .pdf, .md)",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"],
        )

        if uploaded_files and st.button("📥 Build Knowledge Base", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    all_docs = []
                    for uf in uploaded_files:
                        docs = load_from_bytes(uf.read(), uf.name)
                        all_docs.extend(docs)
                        st.caption(f"✓ {uf.name} — {len(docs)} chunks")

                    initialise_rag(docs=all_docs)
                    st.success(f"Knowledge base built: {len(all_docs)} chunks indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process documents: {e}")

        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.rag_pipeline = None
            st.session_state.rag_ready = False
            st.session_state.rag_doc_count = 0
            st.rerun()

        st.divider()

        # ── Chat Controls ──────────────────────────────────
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("💡 Tip: Upload your resume or job description for personalised advice.")


# ─────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────
def render_chat_page():
    st.title(f"{APP_ICON} AI Career Advisor")
    st.caption(
        "Your intelligent guide for tech careers, AI/ML roles, job market insights, "
        "resume tips, and interview preparation."
    )

    # ── Status indicators ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        provider_label = st.session_state.provider.upper()
        st.metric("LLM Provider", provider_label)
    with col2:
        rag_status = "✅ Active" if st.session_state.rag_ready else "⚪ Inactive"
        st.metric("Knowledge Base", rag_status)
    with col3:
        search_status = "✅ On" if st.session_state.web_search_enabled else "⚪ Off"
        st.metric("Web Search", search_status)

    st.divider()

    # ── Load LLM ─────────────────────────────────────────────
    try:
        chat_model = get_llm(
            provider=st.session_state.provider,
            api_key=st.session_state.api_key_override or None,
            model=st.session_state.model_override or None,
        )
    except Exception as e:
        st.error(
            f"❌ Could not initialise LLM: {e}\n\n"
            "Please check your API key in the sidebar or environment variables."
        )
        st.info(
            "**Quick setup:**\n"
            "- Groq (free): https://console.groq.com/keys\n"
            "- OpenAI: https://platform.openai.com/api-keys\n"
            "- Gemini: https://aistudio.google.com/app/apikey"
        )
        return

    # Auto-initialise RAG on first load
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initialising knowledge base..."):
            initialise_rag()

    # ── Display chat history ──────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show source badges if present
            if message.get("used_rag"):
                st.caption("📚 Answer enhanced with knowledge base")
            if message.get("used_search"):
                st.caption("🌐 Answer enhanced with live web search")

    # ── Example prompts (shown when chat is empty) ────────────
    if not st.session_state.messages:
        st.markdown("#### 👋 Try asking:")
        example_cols = st.columns(2)
        examples = [
            "How do I transition into an ML Engineer role?",
            "What skills do I need for a data science job in 2025?",
            "Review my career plan: I'm a backend developer aiming for AI roles.",
            "What are the top AI companies hiring in India right now?",
            "How should I prepare for a Google ML interview?",
            "What's the average salary for a data scientist in Hyderabad?",
        ]
        for i, example in enumerate(examples):
            col = example_cols[i % 2]
            if col.button(f"💬 {example}", key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

    # ── Chat input ────────────────────────────────────────────
    if prompt := st.chat_input("Ask me anything about your tech career..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Determine what context augmentation to use ────────
        rag_context = ""
        web_context = ""
        used_rag = False
        used_search = False

        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            # 1. RAG retrieval
            if st.session_state.rag_ready and st.session_state.rag_pipeline:
                status_placeholder.caption("🔍 Searching knowledge base...")
                try:
                    rag_context = st.session_state.rag_pipeline.retrieve_context(prompt)
                    if rag_context:
                        used_rag = True
                except Exception as e:
                    st.caption(f"⚠️ RAG retrieval failed: {e}")

            # 2. Web search — trigger if enabled and relevant
            if st.session_state.web_search_enabled and should_trigger_search(prompt):
                status_placeholder.caption("🌐 Searching the web...")
                try:
                    web_context, search_ok = get_search_context(prompt, max_results=4)
                    if search_ok and web_context:
                        used_search = True
                except Exception as e:
                    st.caption(f"⚠️ Web search failed: {e}")

            # 3. Build system prompt
            system_prompt = build_system_prompt(
                mode=st.session_state.response_mode,
                rag_context=rag_context,
                web_context=web_context,
            )

            # 4. Stream response
            status_placeholder.caption("✍️ Generating response...")
            try:
                with st.spinner(""):
                    response = get_chat_response(
                        chat_model,
                        st.session_state.messages,
                        system_prompt,
                    )
                status_placeholder.empty()
                st.markdown(response)

                # Source badges
                if used_rag:
                    st.caption("📚 Answer enhanced with knowledge base")
                if used_search:
                    st.caption("🌐 Answer enhanced with live web search")

            except Exception as e:
                status_placeholder.empty()
                response = f"⚠️ Error: {e}"
                st.error(response)

        # Append assistant message with metadata
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "used_rag": used_rag,
                "used_search": used_search,
            }
        )


# ─────────────────────────────────────────────────────────────
# INSTRUCTIONS PAGE
# ─────────────────────────────────────────────────────────────
def render_instructions_page():
    st.title(f"📖 {APP_TITLE} — Setup Guide")

    st.markdown("""
## 🎯 Use Case
This chatbot is an **AI Career Advisor** for technology professionals.
It helps with career transitions, job market insights, resume review, and interview prep —
enhanced by a personal knowledge base (RAG) and live web search.

---

## 🔧 Local Setup

```bash
git clone https://github.com/your-username/ai-career-advisor
cd ai-career-advisor
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Environment Variables

Create a `.env` file or set these in Streamlit Cloud Secrets:

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key        # optional
GOOGLE_API_KEY=your_google_key        # optional
TAVILY_API_KEY=your_tavily_key        # optional — enables premium search
```

**Get API keys:**
- **Groq (free):** https://console.groq.com/keys
- **OpenAI:** https://platform.openai.com/api-keys
- **Gemini:** https://aistudio.google.com/app/apikey
- **Tavily:** https://tavily.com (free tier available)

---

## 📚 Knowledge Base

Upload any `.txt`, `.pdf`, or `.md` documents in the sidebar.

**Recommended documents:**
- Your resume / CV
- Job descriptions you're targeting
- AI/ML learning roadmaps
- Company research notes

---

## ⚡ Response Modes

| Mode | Description |
|------|-------------|
| **Concise** | 3–5 sentence answer, direct and actionable |
| **Detailed** | Structured, comprehensive with examples and steps |

---

## 🌐 Web Search

Automatically activates when your query contains keywords like:
`latest`, `current`, `salary`, `hiring`, `2025`, `trending`

---

## 💬 Example Prompts

- *"How do I transition from backend dev to ML engineer?"*
- *"What are the latest AI job trends in India?"*
- *"Review my career goal: I want to become a data scientist in 6 months."*
- *"What skills does Google look for in AI roles?"*
- *"Compare LLM Engineer vs Data Scientist career paths."*
    """)


# ─────────────────────────────────────────────────────────────
# MAIN NAVIGATION
# ─────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Navigation
    with st.sidebar:
        st.divider()
        page = st.radio(
            "Navigate",
            ["💬 Chat", "📖 Instructions"],
            index=0,
            label_visibility="collapsed",
        )

    if page == "💬 Chat":
        render_chat_page()
    else:
        render_instructions_page()


if __name__ == "__main__":
    main()
