import os
import pickle
import zipfile
import logging
from typing import List, Iterator

# Third-party imports
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain / Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "branham-index"
CHUNK_FILE = "sermon_chunks.pkl"
ZIP_FILE = "sermon_chunks.zip"
DLQ_FILE = "dlq_failed_chunks.pkl"
BATCH_SIZE = 50

# --- LEVEL 3: OBSERVABILITY (LOGGING) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- PHASE 1: PREPARATION & EXTRACTION ---
def ensure_data_ready():
    """Ensures the pickle file exists, unzipping it if necessary."""
    if not os.path.exists(CHUNK_FILE):
        if os.path.exists(ZIP_FILE):
            logger.info(f" Unzipping {ZIP_FILE}...")
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".")
            logger.info(" Unzip complete.")
        else:
            raise FileNotFoundError(f"Missing {CHUNK_FILE} and {ZIP_FILE}. Please upload data.")

def load_chunks_from_disk() -> List[Document]:
    """Loads the parsed documents into memory."""
    ensure_data_ready()
    logger.info(f" Loading {CHUNK_FILE} into memory...")
    with open(CHUNK_FILE, "rb") as f:
        docs = pickle.load(f)
    logger.info(f"🔹 Loaded {len(docs)} raw chunks.")
    return docs

def batch_generator(iterable: List, batch_size: int) -> Iterator[List]:
    """Yields chunks of data to keep memory usage low during processing."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]

# --- PHASE 2: TRANSFORMATION (PURE LOGIC) ---
def apply_monster_check(batch: List[Document]) -> List[Document]:
    """Ensures no single document exceeds Pinecone/Embedding size limits."""
    monster_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    safe_batch = []
    
    for doc in batch:
        if len(doc.page_content) > 30000:
            sub_chunks = monster_splitter.split_documents([doc])
            for j, sub in enumerate(sub_chunks):
                # Safely copy metadata and update paragraph tracking
                sub.metadata = doc.metadata.copy()
                orig_para = doc.metadata.get('paragraph', 'Unknown')
                sub.metadata['paragraph'] = f"{orig_para} (Part {j+1})"
                safe_batch.append(sub)
        else:
            safe_batch.append(doc)
            
    return safe_batch

# --- PHASE 3: LOADING (NETWORK & FAILURES) ---
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True # Passes the error back up if it fails 5 times
)
def upload_batch_to_pinecone(vector_store: PineconeVectorStore, batch: List[Document]):
    """Uploads a batch. Tenacity handles the exponential backoff automatically."""
    vector_store.add_documents(batch)

def run_pipeline():
    """The Orchestrator: Ties the pipeline together."""
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        logger.error("❌ Missing API Keys in environment.")
        return

    # 1. Setup Pinecone Client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
 
    logger.info("Connecting to Vector Store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
     
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # 2. Get Data
    all_docs = load_chunks_from_disk()
    
    # 3. Setup Dead Letter Queue
    dlq_failed_batches = []
    
    # 4. Execute Upload Stream
    logger.info(" Starting Upload Pipeline...")
    
    with tqdm(total=len(all_docs), desc="Uploading") as pbar:
        # We use our generator to stream batches rather than holding massive sub-lists
        for raw_batch in batch_generator(all_docs, BATCH_SIZE):
            
            safe_batch = apply_monster_check(raw_batch)
            
            try:
                upload_batch_to_pinecone(vector_store, safe_batch)
                pbar.update(len(raw_batch))
                
            except Exception as e:
                # Catch unrecoverable failures (DLQ)
                logger.error(f" Batch permanently failed after retries: {e}")
                dlq_failed_batches.extend(safe_batch)
                
    # 5. Pipeline Teardown & Reporting
    if dlq_failed_batches:
        logger.warning(f" Saving {len(dlq_failed_batches)} failed chunks to DLQ: {DLQ_FILE}")
        with open(DLQ_FILE, "wb") as f:
            pickle.dump(dlq_failed_batches, f)
        logger.info("To retry failures later, write a script to load the DLQ file and upload.")
    else:
        logger.info("SUCCESS! 100% of chunks ingested safely.")

if __name__ == "__main__":
    run_pipeline()
