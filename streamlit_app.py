import streamlit as st
import time

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Voice of the Sign",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="auto",
)

# ==============================
# LOAD BACKEND
# ==============================
backend_loaded = False
try:
    from app import get_rag_chain, search_archives
    backend_loaded = True
except Exception as e:
    st.error(f"❌ Backend failed to load:\n\n{e}")

# ==============================
# MESSAGEHUB LINK BUILDER
# ==============================
def messagehub_link(filename: str):
    """
    Example:
    62-0909E In His Presence.pdf
    → https://www.messagehub.info/en/read.do?ref_num=62-0909E
    """
    if not filename:
        return "#"

    name = filename.replace(".pdf", "").replace(".PDF", "").strip()
    code = name.split()[0]  # first token is sermon code
    return f"https://www.messagehub.info/en/read.do?ref_num={code}"


# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

div[data-testid="stChatMessage"][data-test-role="user"] {
    background-color: rgba(128,128,128,0.08);
    border-radius: 20px 20px 5px 20px;
}

div[data-testid="stChatMessage"][data-test-role="assistant"] {
    background-color: rgba(212,175,55,0.05);
    border-left: 4px solid #D4AF37;
    border-radius: 20px 20px 20px 5px;
}

/* Improve markdown spacing */
.markdown-text-container p {
    margin-bottom: 0.9em;
    line-height: 1.7;
}

.markdown-text-container ul {
    margin-left: 1.2em;
}

.markdown-text-container h3 {
    margin-top: 1.2em;
    color: #D4AF37;
}

/* Reference cards */
.quote-card {
    padding: 18px;
    margin-bottom: 14px;
    border-radius: 12px;
    border-left: 5px solid #D4AF37;
}

.quote-meta {
    font-weight: 600;
    margin-bottom: 6px;
}

.quote-meta a {
    color: #D4AF37;
    text-decoration: none;
}

.quote-text {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    line-height: 1.6;
}

@media (prefers-color-scheme: dark) {
    .quote-card {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }
}

@media (prefers-color-scheme: light) {
    .quote-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
}

header, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==============================
# SESSION STATE
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.title("🦅 Controls")
    mode = st.radio("Mode", ["🗣️ Chat", "🔍 Search"], index=0, label_visibility="collapsed")
    st.divider()

    if st.button("🗑️ Clear Screen", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ==============================
# HEADER
# ==============================
col1, col2 = st.columns([1, 14])
with col1:
    st.markdown("# 🦅")
with col2:
    st.markdown("# The 7th Handle" if mode.startswith("🗣️") else "# The Table")

st.divider()

if not backend_loaded:
    st.stop()


# ==============================
# LOAD RAG SYSTEM
# ==============================
@st.cache_resource(show_spinner=False)
def load_chain():
    return get_rag_chain()


# =========================================================
# CHAT MODE
# =========================================================
if mode.startswith("🗣️"):

    # --- Render chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🦅"):
            st.markdown(msg["content"], unsafe_allow_html=False)

            if msg.get("sources"):
                with st.expander("📚 References"):
                    for doc in msg["sources"]:
                        src = doc.metadata.get("source", "")
                        para = doc.metadata.get("paragraph", "")
                        link = messagehub_link(src)
                        st.markdown(f"🔗 [{src} (Para {para})]({link})")

    # --- Input ---
    prompt = st.chat_input("Ask a question...")

    if prompt:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Searching the tapes..."):
            chain = load_chain()
            response = chain.invoke({"question": prompt})

        answer_text = response.get("result", "")
        sources = response.get("source_documents", [])

        # Save assistant message (FULLY formed)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources
        })

        # Force rerender so references appear immediately
        st.rerun()


# =========================================================
# SEARCH MODE
# =========================================================
else:
    query = st.chat_input("Search for a keyword...")

    if query:
        st.subheader(f"Results for: “{query}”")

        with st.spinner("Scanning archives..."):
            docs, debug_log = search_archives(query)

        with st.expander("🛠 Debug Info"):
            for line in debug_log:
                st.write(line)

        if not docs:
            st.info("No exact records found.")
        else:
            for doc in docs:
                src = doc.metadata.get("source", "")
                para = doc.metadata.get("paragraph", "")
                link = messagehub_link(src)

                st.markdown(f"""
                <div class="quote-card">
                    <div class="quote-meta">
                        <a href="{link}" target="_blank">
                            📼 {src} (Para {para})
                        </a>
                    </div>
                    <div class="quote-text">
                        "{doc.page_content}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
