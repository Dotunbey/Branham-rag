# 🦅 Branham-RAG

**A NotebookLM‑Grade Retrieval‑Augmented AI for Large Sermon Archives**

---

## 📌 Overview

**Branham-RAG** is a production‑ready **Retrieval‑Augmented Generation (RAG)** system designed to deliver **accurate, reference‑grounded answers and summaries** over a large historical sermon corpus.

Unlike typical chatbot demos, this system emphasizes:

* Deterministic parsing and metadata extraction
* Identity‑preserving chunking
* Hybrid retrieval (keyword + semantic + series‑aware)
* Doctrine‑faithful, structured generation
* Verifiable references that link directly to original sermon sources

The architecture and design philosophy are intentionally aligned with systems like **NotebookLM**, prioritizing **accuracy, traceability, and interpretability** over raw fluency.

---

## 🎯 Key Features

### 🔹 Robust PDF Ingestion Pipeline

* Filename‑based metadata extraction (date code, title)
* Visual header and page‑noise removal
* Paragraph‑accurate parsing aligned with original transcripts
* Page start/end tracking per paragraph
* Identity‑preserving handling of very large (“monster”) paragraphs
* Incremental ingestion (new sermons can be added without reprocessing old data)

---

### 🔹 Deterministic Metadata & Citation Safety

Every text chunk retains stable metadata, including:

* Sermon filename
* Date code (e.g. `62‑0909E`)
* Paragraph number
* Page range
* Stable chunk identity

This ensures:

* No duplicate vectors
* No broken references
* Predictable citation behavior
* Safe re‑ingestion and resume support

---

### 🔹 Hybrid Retrieval Engine (NotebookLM‑Style)

The retriever combines **four complementary strategies**:

1. **Explicit sermon targeting**
   Example: *“Summarize In His Presence”*

2. **Series‑aware retrieval**
   Correctly prioritizes canonical series (e.g. the 1963 Seven Seals sermons)

3. **Local keyword & BM25 ranking**
   High‑recall matching for exact phrases and terminology

4. **Cloud semantic search (Pinecone)**
   High‑precision embedding‑based retrieval

Results are **deduplicated, ranked, and merged deterministically** before being passed to the LLM.

---

### 🔹 Doctrine‑Safe Prompt Engineering

The system prompt enforces:

* Faithful paraphrasing (no invented doctrine)
* Clear structural output (headings and bullet points)
* Explicit explanation of symbols
* Calm, historical preaching tone
* Explicit uncertainty when the source text is silent
* Strict avoidance of hallucinated citations

This significantly reduces misrepresentation risk and improves trustworthiness.

---

### 🔹 Streamlit UI (Production‑Focused UX)

* Chat mode for conversational Q&A
* Search mode for direct paragraph discovery
* Immediate, clickable references
* Mobile‑friendly sidebar behavior
* Structured output rendering (headings, bullets, paragraphs preserved)
* Deep‑linking to original sermons on MessageHub

---

## 🧠 Architecture Overview

```
PDFs
 └── Ingestion Engine
      ├── Metadata Extraction
      ├── Page & Paragraph Parsing
      ├── Chunk Identity Control
      └── Incremental Cache (Pickle)

Chunks
 ├── Local Search (BM25)
 ├── Vector Store (Pinecone)
 └── Hybrid Retriever
      └── LLM (Gemini)

Streamlit UI
 ├── Chat Mode
 ├── Search Mode
 └── Reference Linking
```

---

## 🛠️ Tech Stack

| Layer               | Technology           |
| ------------------- | -------------------- |
| Language            | Python               |
| UI                  | Streamlit            |
| PDF Processing      | PyMuPDF              |
| Retrieval Framework | LangChain            |
| Vector Database     | Pinecone             |
| Embeddings          | Google Generative AI |
| LLM                 | Gemini               |
| Keyword Search      | BM25                 |
| Caching             | Pickle               |
| Deployment          | Streamlit Cloud      |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/voice-of-the-sign.git
cd voice-of-the-sign
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

---

## 📥 Ingestion Workflow

```bash
python ingest_local.py
```

Supports:

* Resume after failure
* Incremental updates
* Safe re‑runs
* No duplicate vectors

---

## 🚀 Running the App

```bash
streamlit run streamlit_app.py
```

---

## 🔗 Reference Linking

All references open directly to the original sermon on MessageHub:

```
https://www.messagehub.info/en/read.do?ref_num=62-0909E
```

This allows users to instantly verify AI responses against primary sources.

---

## 🧪 Example Queries

* *“Summarize In His Presence”*
* *“What does the white horse represent in the Seven Seals?”*
* *“Explain the Laodicean Church Age”*
* *“What did he say about justification?”*

---

## 🧩 Design Philosophy

This project was intentionally built to demonstrate:

* Data engineering rigor
* LLM safety awareness
* Explainable AI principles
* Scalable ingestion design
* Professional RAG architecture

It is **not** a toy chatbot, but a **research‑grade retrieval system** designed around real‑world constraints.

---

## 👤 Author

**Aina Adoption Oluwasomidotun**
AI Engineer | Backend Engineer | RAG Systems Builder

Focused on:

* Retrieval‑Augmented Generation
* Explainable AI
* Scalable ingestion pipelines
* LLM safety and grounding

---

## 📎 License

This project is provided for demonstration and educational purposes.

---

### ✅ Recruiter Note

This project demonstrates **end‑to‑end ownership** of:

* Data ingestion
* Indexing and retrieval
* Prompt engineering
* UI integration
* Deployment considerations

It reflects the architectural thinking expected in **production‑grade AI systems** at serious engineering organizations.
