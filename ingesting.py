import os
import re
import pickle
import time
import zipfile
import logging
from pathlib import Path
from typing import List, Iterator

import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "branham-index"
SOURCE_DIRECTORY = Path("./sermons") # Upgraded to Pathlib
CHUNK_FILE = Path("sermon_chunks.pkl")
DLQ_FILE = Path("dlq_failed_chunks.pkl")
BATCH_SIZE = 50

# --- OBSERVABILITY ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# PHASE 1: EXTRACT (Reading the PDFs)
# ==========================================

def process_file_adaptive(file_path: Path) -> List[Document]:
    """
    INTELLIGENT PARSER: Scans text for paragraph numbers.
    Uses strict splitting if numbered, standard chunking if not.
    """
    try:
        doc = fitz.open(file_path)
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF {file_path.name}: {e}")
        return []

    lines = full_text.split('\n')
    para_pattern = re.compile(r'^\s*(E-\d+|\d+)(?:\.|:)?\s+')
    
    # Strategy Check
    number_matches = sum(1 for line in lines if para_pattern.match(line))
    is_numbered_sermon = number_matches > 5

    documents = []

    if is_numbered_sermon:
        # STRATEGY A: PARAGRAPH SPLITTING
        current_para_num = "Intro"
        current_text_buffer = []

        for line in lines:
            line = line.strip()
            if not line: continue

            match = para_pattern.match(line)
            if match:
                # Save Previous
                if current_text_buffer:
                    combined_text = " ".join(current_text_buffer)
                    if len(combined_text) > 20:
                        documents.append(Document(
                            page_content=combined_text,
                            metadata={"source": file_path.name, "paragraph": current_para_num}
                        ))
                # Start New
                current_para_num = match.group(1)
                current_text_buffer = [line]
            else:
                current_text_buffer.append(line)
        
        # Save Tail
        if current_text_buffer:
            documents.append(Document(
                page_content=" ".join(current_text_buffer),
                metadata={"source": file_path.name, "paragraph": current_para_num}
            ))
            
    else:
        # STRATEGY B: FALLBACK CHUNKING
        raw_doc = Document(page_content=full_text, metadata={"source": file_path.name})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents([raw_doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["paragraph"] = f"Unnumbered (Chunk {i+1})"
            documents.append(chunk)

    return documents

def extract_all_pdfs() -> List[Document]:
    """Finds all PDFs in the source directory and extracts them."""
    if not SOURCE_DIRECTORY.exists():
        logger.warning(f"Source directory {SOURCE_DIRECTORY} not found. Returning empty list.")
        return []

    pdf_files = list(SOURCE_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        return []

    logger.info(f"📂 Found {len(pdf_files)} PDFs. Starting Adaptive Extraction...")
    all_docs = []
    
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        docs = process_file_adaptive(pdf_file)
        all_docs.extend(docs)
        
    return all_docs


# ==========================================
# PHASE 2: TRANSFORM (Pre-Processing)
# ==========================================

def apply_monster_check(batch: List[Document]) -> List[Document]:
    """Ensures no single document exceeds Pinecone size limits."""
    monster_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    safe_batch = []
    
    for doc in batch:
        if len(doc.page_content) > 30000:
            sub_chunks = monster_splitter.split_documents([doc])
            for j, sub in enumerate(sub_chunks):
                sub.metadata = doc.metadata.copy()
                orig_para = doc.metadata.get('paragraph', 'Unknown')
                sub.metadata['paragraph'] = f"{orig_para} (Part {j+1})"
                safe_batch.append(sub)
        else:
            safe_batch.append(doc)
            
    return safe_batch

def batch_generator(iterable: List, batch_size: int) -> Iterator[List]:
    """Yields chunks of data to keep memory usage low."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


# ==========================================
# PHASE 3: LOAD (Upload to Pinecone)
# ==========================================

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=10), reraise=True)
def upload_batch_to_pinecone(vector_store: PineconeVectorStore, batch: List[Document]):
    """Uploads a batch with automatic exponential backoff for network errors."""
    vector_store.add_documents(batch)

def run_pipeline():
    """The Orchestrator: Ties the ETL pipeline together."""
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        logger.error(" Missing API Keys. Halting.")
        return

    # --- 1. RESOLVE DATA STATE (Extract or Load from Cache) ---
    all_docs = []
    if CHUNK_FILE.exists():
        choice = input(f" Found {CHUNK_FILE}. Skip PDF reading and use cached data? (y/n): ")
        if choice.lower() == 'y':
            logger.info(" Loading cached chunks from disk...")
            with open(CHUNK_FILE, "rb") as f:
                all_docs = pickle.load(f)
    
    if not all_docs:
        all_docs = extract_all_pdfs()
        if all_docs:
            logger.info("Saving newly extracted chunks to disk cache...")
            with open(CHUNK_FILE, "wb") as f:
                pickle.dump(all_docs, f)
        else:
            logger.error(" No data extracted. Halting.")
            return

    logger.info(f"🔹 Ready to process {len(all_docs)} total chunks.")

    # --- 2. SETUP VECTOR STORE ---
    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # --- 3. DETERMINE START POINT (Resume Capability) ---
    start_index = 0
    start_input = input("Enter start index (Press Enter for 0, or type a number to resume): ")
    if start_input.isdigit():
        start_index = int(start_input)

    docs_to_upload = all_docs[start_index:]
    dlq_failed_batches = []

    # --- 4. EXECUTE UPLOAD STREAM ---
    logger.info(f"🚀 Uploading starting at index {start_index}...")
    
    with tqdm(total=len(docs_to_upload), desc="Uploading", initial=start_index) as pbar:
        for raw_batch in batch_generator(docs_to_upload, BATCH_SIZE):
            
            # Transform
            safe_batch = apply_monster_check(raw_batch)
            
            # Load
            try:
                upload_batch_to_pinecone(vector_store, safe_batch)
                pbar.update(len(raw_batch))
            except Exception as e:
                logger.error(f" Batch permanently failed after retries: {e}")
                dlq_failed_batches.extend(safe_batch)
                
    # --- 5. TEARDOWN & REPORTING ---
    if dlq_failed_batches:
        logger.warning(f"Saving {len(dlq_failed_batches)} failed chunks to {DLQ_FILE}")
        with open(DLQ_FILE, "wb") as f:
            pickle.dump(dlq_failed_batches, f)
    else:
        logger.info(" SUCCESS! Database is fully updated.")

if __name__ == "__main__":
    run_pipeline()