# Branham RAG Chatbot

An AI simulation that serves as an assistant to William Branham using a local library of sermon PDFs. It uses Hybrid Retrieval (Semantic + Keyword) and Google Gemini to generate responses strictly grounded in the text.

## Prerequisites

1.  **Python 3.9+** installed.
2.  **Google API Key** (Get it from [Google AI Studio](https://aistudio.google.com/)).

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data:**
    * Create a folder named `sermons` in the root directory: `mkdir sermons`
    * Place your ~1,200 PDF sermons inside `./sermons/`.

3.  **Build the Index:**
    * Run the indexing script. This parses the PDFs, creates embeddings, and builds the database.
    * *Note: This runs once. Re-run only if you add new PDFs.*
    ```bash
    python 01_build_index.py
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Usage

* **API Key:** When the app launches, enter your Google API Key in the sidebar (or set `GOOGLE_API_KEY` as an environment variable).
* **Model Switching:** Toggle between `gemini-1.5-flash` (Fast) and `gemini-1.5-pro` (High Reasoning) in the sidebar.
* **Citations:** Click "View Source Sermons" under every response to see exactly which PDF and text chunk the AI used.

## Architecture

* **Extraction:** PyMuPDF (`fitz`)
* **Chunking:** Recursive Character Splitter (1000 tokens, 200 overlap)
* **Embeddings:** `sentence-transformers/all-mpnet-base-v2`
* **Vector Store:** FAISS
* **Retrieval:** Ensemble (70% FAISS Semantic + 30% BM25 Keyword)
* **LLM:** Google Gemini via LangChain
