# RAG Ingestion Pipeline Guide & Concepts

This guide provides a comprehensive explanation of the architecture, key concepts, and advantages of the modern RAG (Retrieval-Augmented Generation) ingestion pipeline implemented in [`02_rag_implementation/rag_ingestion.py`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/rag_ingestion.py).

---

## 1. Overview of the Ingestion Pipeline

The ingestion pipeline processes raw text documents (such as legal texts or acts) and transforms them into searchable vector representations stored in Pinecone:

```
[ Raw Document (.txt) ]
          │
          ▼  1. Load
[ Document Object ] (Text + Metadata)
          │
          ▼  2. Token-Based Chunking & Overlap
[ Chunked Documents ] (Preserved & Enriched Metadata)
          │
          ▼  3. Embedding Generation
[ Vector Embeddings ] (Google Generative AI 1536d)
          │
          ▼  4. Batch Upload with Explicit IDs
[ Pinecone Vector Store ] (Deterministic IDs & Indexing)
```

---

## 2. Core Concepts & Advantages of Made Changes

### 1. Fail-Fast Environment Validation (`validate_env`)

- **Concept**: Before performing network or disk operations, validate that all required API keys (`PINECONE_API_KEY`, `GOOGLE_API_KEY`) and configurations (`GST_ACT_INDEX_NAME`) exist.
- **Advantages**:
  - **Fail-Fast**: Prevents partial document processing or silent failures mid-ingestion.
  - **Developer Experience**: Clear, actionable error messages instruct what `.env` keys are missing.

---

### 2. Token-Based Chunking (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`)

- **Concept**: Splitting text based on BPE (Byte-Pair Encoding) tokens (`cl100k_base`) rather than raw character counts.
- **Advantages**:
  - **Aligned with LLM Context Limits**: LLMs have token limits, not character limits. Measuring chunks in tokens prevents context overflow.
  - **Consistent Density**: Characters vary in length (punctuation vs long words), while tokens represent semantic units.
  - **Recursive Hierarchical Splitting**: Tries splitting by double newlines (`\n\n`), single newlines (`\n`), spaces (` `), and characters (`""`) in sequence to preserve sentence and paragraph structure.

---

### 3. Chunk Overlap (`chunk_overlap=50`)

- **Concept**: Retaining a sliding window of tokens between consecutive chunks.
- **Advantages**:
  - **Boundary Context Retention**: Sentences split across chunk boundaries retain full semantic context.
  - **Prevents Information Loss**: Key terms located at chunk edges are embedded with their surrounding context in at least one chunk.

---

### 4. Metadata Preservation & Enrichment

- **Concept**: Retaining original document properties (`source`) while attaching chunk position data (`chunk_index`, `total_chunks`).
- **Advantages**:
  - **Source Attribution**: RAG applications can display exact source file names for generated answers.
  - **Sequence Awareness**: Allows re-assembling or querying adjacent chunks during retrieval.

---

### 5. Explicit & Deterministic Vector IDs

- **Concept**: Generating predictable IDs for each vector (e.g., `CGST-Act-Updated-30092020-chunk-0`).
- **Advantages**:
  - **Idempotency**: Running the script multiple times updates existing vectors instead of duplicating them.
  - **Deletion & Upsert Control**: Enables easy targeted deletion or updating of specific document chunks.

---

### 6. Batch Vector Uploads (`batch_size=100`)

- **Concept**: Uploading vectors in chunks of 100 rather than sending hundreds of vectors in a single HTTP request.
- **Advantages**:
  - **Payload Optimization**: Avoids HTTP payload size limits (413 errors) and API timeouts.
  - **Resilience & Progress Visibility**: Displays clear batch-by-batch progress.

---

### 7. Modern Pinecone Vector Store Pattern

- **Concept**: Instantiating `PineconeVectorStore(index_name=..., embedding=...)` directly instead of using convenience constructors like `from_documents`.
- **Advantages**:
  - **Explicit Control**: Separation of vector store initialization from document upserting logic.
  - **Flexibility**: Facilitates reusing vector store connections across different pipeline functions.

---

## 3. Code Walkthrough (`02_rag_implementation/rag_ingestion.py`)

Below is the complete logic breakdown of [`02_rag_implementation/rag_ingestion.py`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/rag_ingestion.py):

```python
# 1. Environment Validation
def validate_env() -> dict[str, str]:
    required_vars = ["PINECONE_API_KEY", "GOOGLE_API_KEY", "GST_ACT_INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return {var: os.environ[var] for var in required_vars}

# 2. Document Loading
def load_document(file_path: str) -> list[Document]:
    loader = TextLoader(file_path)
    return loader.load()

# 3. Token-Based Chunking with Overlap & Metadata
def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks

# 4. Batch Ingestion with Explicit IDs
def ingest_to_vectorstore(chunks: list[Document], index_name: str, batch_size: int = 100):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=1536,
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    all_ids = [f"{Path(c.metadata.get('source', 'doc')).stem}-chunk-{c.metadata['chunk_index']}" for c in chunks]

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
```
