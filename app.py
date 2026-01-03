import os
import pickle
from typing import List, Dict, Set
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()

# ===============================
# CONFIG
# ===============================
INDEX_NAME = "branham-index"
CHUNKS_FILE = "sermon_chunks.pkl"

# ===============================
# CANONICAL SERIES
# ===============================
SEVEN_SEALS_CANON = [
    "63-0317E The Breach Between The Church Ages And The Seven Seals.pdf",
    "63-0317M God Hiding Himself In Simplicity, Then Revealing Himself In The Same.pdf",
    "63-0318 The First Seal.pdf",
    "63-0319 The Second Seal.pdf",
    "63-0320 The Third Seal.pdf",
    "63-0321 The Fourth Seal.pdf",
    "63-0322 The Fifth Seal.pdf",
    "63-0323 The Sixth Seal.pdf",
    "63-0324E The Seventh Seal.pdf",
    "63-0324M Questions And Answers On The Seals.pdf",
]

SERIES_GROUPS = {
    "seven seals": SEVEN_SEALS_CANON,
}

# ===============================
# HELPERS
# ===============================
def normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip()


def load_chunks() -> List[Document]:
    if not os.path.exists(CHUNKS_FILE):
        return []
    with open(CHUNKS_FILE, "rb") as f:
        return pickle.load(f)


def extract_date_code(filename: str) -> str:
    """
    Assumes filenames start with NN-NNNNE
    Example: 62-0909E In His Presence.pdf
    """
    return filename.split()[0].replace(".pdf", "")


def messagehub_link(filename: str) -> str:
    code = extract_date_code(filename)
    return f"https://www.messagehub.info/en/read.do?ref_num={code}"



import re

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "and", "to", "for", "with", "by"
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sermon_title(filename: str) -> str:
    """
    '62-0909E In His Presence.pdf' → 'in his presence'
    """
    name = filename.replace(".pdf", "").replace(".PDF", "")

    parts = name.split(" ", 1)
    if len(parts) == 2 and "-" in parts[0]:
        name = parts[1]

    return normalize_text(name)


def tokenize_meaningful(text: str) -> set:
    return {
        w for w in normalize_text(text).split()
        if w not in STOPWORDS and len(w) > 2
    }


def sermon_title_matches(user_query: str, filename: str) -> bool:
    """
    Match only if ALL meaningful title words exist in user query.
    Prevents partial matches like 'presence'.
    """
    title_tokens = tokenize_meaningful(extract_sermon_title(filename))
    query_tokens = tokenize_meaningful(user_query)

    if not title_tokens:
        return False

    return title_tokens.issubset(query_tokens)

# ===============================
# RETRIEVER
# ===============================
class BranhamRetriever(BaseRetriever):
    """
    NotebookLM-style hybrid retriever:
    - local priority
    - semantic fallback
    - series-aware
    - safe + deduplicated
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        query_clean = normalize(query)
        chunks = load_chunks()
        results: List[Document] = []
        seen = set()

        # -------------------------------------------------
        # Detect sermon reference (date code)
        # -------------------------------------------------
        explicit_sermon = None
        for token in query.split():
            if "-" in token and len(token) >= 7:
                explicit_sermon = token.upper()
                break

        # -------------------------------------------------
        # Detect series
        # -------------------------------------------------
        target_titles = []
        is_series = False

        for key, titles in SERIES_GROUPS.items():
            if key in query_clean:
                target_titles = titles
                is_series = True
                break

        # -------------------------------------------------
        # SERMON-TARGETED SEARCH
        # -------------------------------------------------
        if explicit_sermon:
            for d in chunks:
                src = normalize(d.metadata.get("source", ""))
                if sermon_title_matches(explicit_sermon, src):
                    key = d.page_content[:120]
                    if key not in seen:
                        results.append(d)
                        seen.add(key)

        # -------------------------------------------------
        # SERIES SEARCH
        # -------------------------------------------------
        elif target_titles:
            for d in chunks:
                src = normalize(d.metadata.get("source", ""))
                if sermon_title_matches(title, src):
                    key = d.page_content[:120]
                    if key not in seen:
                        results.append(d)
                        seen.add(key)
        
        # -------------------------------------------------
        # KEYWORD SEARCH (LOCAL)
        # -------------------------------------------------
        if len(results) < 25:
            bm25 = BM25Retriever.from_documents(chunks)
            bm25.k = 60
            for d in bm25.invoke(query):
                key = d.page_content[:120]
                if key not in seen:
                    results.append(d)
                    seen.add(key)

        # -------------------------------------------------
        # VECTOR SEARCH (PINECONE)
        # -------------------------------------------------
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            store = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=embeddings
            )

            vec_docs = store.as_retriever(search_kwargs={"k": 30}).invoke(query)
            for d in vec_docs:
                key = d.page_content[:120]
                if key not in seen:
                    results.append(d)
                    seen.add(key)

        except Exception:
            pass

        return results


# ===============================
# PROMPT
# ===============================
PROMPT_TEMPLATE = """
You are William Marrion Branham, speaking carefully as a teacher and evangelist.

RULES:
- Be faithful to the sermons provided.
- Do NOT invent doctrine.
- avoid 
- If something is not clearly stated in the text, say so.
- Use calm 1950s preaching tone.
- Be structured and clear.
- Use headings and bullet points.
- Explain symbols plainly.
- Prefer paraphrase, but preserve meaning.
- Avoid citations like (54) or paragraph numbers.
- Ignore tape noise or filler language.
- If a question asks for a sermon summary, summarize only that sermon.
- If the question references the Seven Seals, prioritize the 1963 series.

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context_str", "question"],
)

# ===============================
# PUBLIC API
# ===============================
def get_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.25,
        convert_system_message_to_human=True,
    )

    retriever = BranhamRetriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_variable_name": "context_str",
        },
        input_key="question",
    )

    return chain


def search_archives(query: str):
    """
    Used by Search mode only.
    Returns (documents, debug_log)
    """
    debug = []
    docs = []
    seen = set()

    chunks = load_chunks()
    query_clean = normalize(query)

    # Keyword search
    for d in chunks:
        if query_clean in d.page_content.lower():
            key = d.page_content[:120]
            if key not in seen:
                docs.append(d)
                seen.add(key)

    debug.append(f"Keyword hits: {len(docs)}")

    # Fallback BM25
    if len(docs) < 20:
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 50
        for d in bm25.invoke(query):
            key = d.page_content[:120]
            if key not in seen:
                docs.append(d)
                seen.add(key)

    debug.append(f"Total results: {len(docs)}")

    return docs, debug
