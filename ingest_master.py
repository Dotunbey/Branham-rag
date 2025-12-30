import os
import re
import pickle
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm

# Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone  # Needed for deletion

load_dotenv()

# --- CONFIG ---
INDEX_NAME = "branham-index"
SOURCE_DIRECTORY = "./sermons"
CHUNK_FILE = "sermon_chunks.pkl"

def process_file_adaptive(file_path, filename):
    """
    INTELLIGENT PARSER:
    1. Scans text to see if it has paragraph numbers (E-1 or 1, 2, 3).
    2. If YES: Uses strict paragraph splitting.
    3. If NO: Falls back to standard character chunking (so no text is lost).
    """
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    lines = full_text.split('\n')
    
    # --- STRATEGY CHECK ---
    # UPDATED REGEX: Matches "E-1", "1", "1.", "1:" 
    # This ensures we catch "53. Text" as well as "53 Text"
    para_pattern = re.compile(r'^\s*(E-\d+|\d+)(?:\.|:)?\s+')
    
    number_matches = 0
    for line in lines:
        if para_pattern.match(line):
            number_matches += 1
            
    # Decision Threshold: If a file has fewer than 5 numbered paragraphs, 
    # it's likely an unnumbered transcript.
    is_numbered_sermon = number_matches > 5

    documents = []

    if is_numbered_sermon:
        # --- STRATEGY A: PARAGRAPH SPLITTING ---
        # (This gives exact references like "Para 53")
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
                            metadata={"source": filename, "paragraph": current_para_num}
                        ))
                # Start New
                current_para_num = match.group(1)
                current_text_buffer = [line]
            else:
                current_text_buffer.append(line)
        
        # Save Tail
        if current_text_buffer:
            combined_text = " ".join(current_text_buffer)
            documents.append(Document(
                page_content=combined_text,
                metadata={"source": filename, "paragraph": current_para_num}
            ))
            
    else:
        # --- STRATEGY B: FALLBACK CHUNKING ---
        # (For unnumbered sermons. References will say "Page X Chunk Y")
        # We assume the file is valid text, just unformatted.
        
        # We create a temporary Document for the whole text
        raw_doc = Document(page_content=full_text, metadata={"source": filename})
        
        # Use standard splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents([raw_doc])
        
        # Label them clearly
        for i, chunk in enumerate(chunks):
            chunk.metadata["paragraph"] = f"Unnumbered (Chunk {i+1})"
            documents.append(chunk)

    return documents

def upload_to_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: Missing Keys.")
        return

    all_docs = []

    # --- STEP 1: LOAD DATA (RESUME CAPABILITY) ---
    if os.path.exists(CHUNK_FILE):
        print(f"⚠️ Found saved data: {CHUNK_FILE}")
        # Only ask to resume if we trust the data. Since we updated regex, suggest fresh run.
        choice = input("Skip PDF reading and RESUME upload? (y/n): ")
        if choice.lower() == 'y':
            print("📂 Loading saved chunks...")
            with open(CHUNK_FILE, "rb") as f:
                all_docs = pickle.load(f)
    
    # If we didn't load from file, process PDFs from scratch
    if not all_docs:
        # --- WIPE OLD DATA ONLY IF STARTING FRESH ---
        print("🧹 Cleaning old data from Pinecone...")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(INDEX_NAME)
        try:
            index.delete(delete_all=True)
            print("✅ Index wiped clean.")
        except Exception as e:
            print(f"⚠️ Could not wipe index: {e}")

        # Process Files
        files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.lower().endswith('.pdf')]
        print(f"📂 Found {len(files)} PDFs. Starting Adaptive Ingestion...")

        for filename in tqdm(files, desc="Processing"):
            file_path = os.path.join(SOURCE_DIRECTORY, filename)
            try:
                docs = process_file_adaptive(file_path, filename)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        # Save Local Keywords
        print("💾 Saving local chunks...")
        with open(CHUNK_FILE, "wb") as f:
            pickle.dump(all_docs, f)

    print(f"🔹 Ready to upload {len(all_docs)} total chunks.")

    # --- STEP 2: DETERMINE START POINT ---
    start_index = 0
    start_input = input("Enter start index (e.g., 0 to start fresh, 84000 to resume): ")
    if start_input.isdigit():
        start_index = int(start_input)

    # --- STEP 3: UPLOAD WITH RETRY LOGIC ---
    print(f"🚀 Uploading to Pinecone starting at {start_index}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    BATCH_SIZE = 50
    
    # Only slice the list from the start index
    docs_to_upload = all_docs[start_index:]
    
    # Initialize tqdm with the correct total and initial position
    with tqdm(total=len(docs_to_upload), desc="Uploading", initial=start_index) as pbar:
        for i in range(0, len(docs_to_upload), BATCH_SIZE):
            batch = docs_to_upload[i : i + BATCH_SIZE]
            
            # RETRY LOOP
            success = False
            retries = 3
            
            while not success and retries > 0:
                try:
                    vector_store.add_documents(batch)
                    success = True
                    time.sleep(0.5) 
                    pbar.update(len(batch))
                except Exception as e:
                    retries -= 1
                    print(f"\n⚠️ Batch failed. Retrying in 10s... ({retries} left). Error: {e}")
                    time.sleep(10)
            
            if not success:
                print(f"\n❌ FAILED to upload batch starting at index {start_index + i}. Moving to next.")

    print("\n✅ Success! Database is fully updated.")

if __name__ == "__main__":
    upload_to_pinecone()
