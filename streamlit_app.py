import streamlit as st
import time
import os

# Import the backend logic from app.py
# Make sure app.py is in the same folder!
try:
    from app import get_rag_chain
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import app.py. Make sure the file exists and libraries are installed. Details: {e}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Voice of the Sign", 
    page_icon="🦅", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Chat Bubble Styling */
    .stChatMessage { 
        border-radius: 12px; 
        border: 1px solid #E0E0E0; 
        padding: 10px;
    }
    div[data-testid="stChatMessage"]:nth-child(even) { 
        background-color: #FFFFFF; 
        border-left: 5px solid #8B5E3C; /* Brown accent for Branham theme */
    }
    div[data-testid="stChatMessage"]:nth-child(odd) { 
        background-color: #F8F9FA; 
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 Voice of the Sign")
    st.info("This AI allows you to search the Message using Hybrid Technology (Vector + Keywords).")
    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "God bless you. I am here to search the tapes for you. What is on your heart?"}
    ]

# --- LOAD THE BRAIN (With Spinner) ---
# We use cache_resource so it only connects to Pinecone once
@st.cache_resource(show_spinner=False)
def load_chain():
    return get_rag_chain()

# Placeholder for the loading state
if "chain" not in st.session_state:
    with st.spinner("Connecting to Pinecone & Unzipping Keyword Data... (This takes 10s)"):
        try:
            st.session_state.chain = load_chain()
        except Exception as e:
            st.error(f"Failed to load system: {e}")
            st.stop()

chain = st.session_state.chain

# --- DISPLAY CHAT HISTORY ---
st.title("🦅 THE 7TH HANDLE")
st.caption("Interactive Archive • Powered by Gemini & Pinecone")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="📖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        
        # Display Sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sermon References"):
                for src in msg["sources"]:
                    st.markdown(f"- *{src}*")

# --- USER INPUT HANDLING ---
if prompt := st.chat_input("Ask a question (e.g., 'What is the Third Pull?')..."):
    
    # 1. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Searching the archives..."):
            try:
                # Run the RAG Chain
                response = chain.invoke({"query": prompt})
                result_text = response['result']
                
                # --- EXTRACT REFERENCES ---
                source_docs = response.get('source_documents', [])
                formatted_sources = []
                
                for doc in source_docs:
                    # Get metadata
                    filename = doc.metadata.get('source', 'Unknown Sermon')
                    paragraph = doc.metadata.get('paragraph', 'Intro')
                    
                    # Format: "65-1127.pdf (Para E-5)"
                    formatted_sources.append(f"{filename} (Para {paragraph})")
                
                # Remove duplicates while keeping order
                unique_sources = list(dict.fromkeys(formatted_sources))

                # --- TYPING EFFECT ---
                placeholder = st.empty()
                full_response = ""
                # Simulate typing
                for chunk in result_text.split():
                    full_response += chunk + " "
                    time.sleep(0.04)
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                # --- SAVE TO HISTORY ---
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": unique_sources
                })
                
                # --- SHOW SOURCES IMMEDIATELY ---
                if unique_sources:
                    with st.expander("📚 Sermon References"):
                        for src in unique_sources:
                            st.markdown(f"- *{src}*")

            except Exception as e:
                st.error(f"An error occurred: {e}")
