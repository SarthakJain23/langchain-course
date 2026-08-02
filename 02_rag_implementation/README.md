# Chapter 2: RAG Implementation, Ingestion, Agent & LCEL Retrieval

This chapter covers building a complete Retrieval-Augmented Generation (RAG) system using LangChain, Google Generative AI Embeddings (`models/gemini-embedding-001`), Pinecone Vector Store, and Gemini (`gemini-2.5-flash`).

## Files Included

- **[`rag_ingestion.py`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/rag_ingestion.py)**: Ingestion script with fail-fast env validation, token-based chunking (`cl100k_base`), metadata enrichment, batching, and explicit vector IDs.
- **[`rag_agent.py`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/rag_agent.py)**: Tool-calling Agent implementation using `create_agent`.
- **[`rag_lcel.py`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/rag_lcel.py)**: Declarative functional RAG chain using **LCEL** (`retriever | prompt | llm | StrOutputParser()`).
- **[`LCEL_AND_MMR_EXPLAINED.md`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/LCEL_AND_MMR_EXPLAINED.md)**: In-depth beginner guide explaining LCEL parallel runnable dictionaries and MMR retrieval parameters.
- **[`RAG_INGESTION_GUIDE.md`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/RAG_INGESTION_GUIDE.md)**: Technical architecture document detailing the ingestion pipeline design principles.
- **[`CGST-Act-Updated-30092020.txt`](file:///Users/sarthakjain/Desktop/langchain-course/02_rag_implementation/CGST-Act-Updated-30092020.txt)**: Sample dataset (CGST Act document) used for vector indexing.

## Required Environment Variables

Ensure your root `.env` file includes:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
GST_ACT_INDEX_NAME=your_pinecone_index_name
```

## How to Run

### 1. Ingest Documents into Pinecone

```bash
python 02_rag_implementation/rag_ingestion.py
```

### 2. Query via Tool-Calling Agent

```bash
python 02_rag_implementation/rag_agent.py "What is the penalty for late tax filing under CGST?"
```

### 3. Query via Declarative LCEL Chain

```bash
python 02_rag_implementation/rag_lcel.py "What is the penalty for late tax filing under CGST?"
```
