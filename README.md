# LangChain Course

A modular repository organized by course chapters for learning LangChain, agents, tool calling, and RAG pipelines.

## Project Structure

All chapters share the root environment configuration (`.env`) and dependencies (`pyproject.toml` / `.venv`), keeping code clean and modular without duplicate setups.

```
langchain-course/
├── .env                              # Shared environment variables across all chapters
├── pyproject.toml                    # Shared python dependencies
├── 01_agents_and_tools/              # Chapter 1: Agents, Structured Outputs & Tool Calling
│   ├── main.py                       # Agent with structured output (AgentResponse, Source)
│   ├── tool_calling.py               # Custom tool calling agent
│   └── README.md                     # Chapter 1 guide
└── 02_rag_implementation/            # Chapter 2: RAG Ingestion Pipeline & Agent Retrieval
    ├── CGST-Act-Updated-30092020.txt # Sample text dataset
    ├── RAG_INGESTION_GUIDE.md        # Technical guide & architecture breakdown
    ├── rag_ingestion.py              # Vector store ingestion pipeline script
    ├── rag_agent.py                  # Pinecone + Gemini RAG query agent script
    └── README.md                     # Chapter 2 guide
```

## Setup & Environment

1. **Install Dependencies**:
   ```bash
   uv sync
   # or
   pip install -e .
   ```

2. **Configure Environment Variables**:
   Create or edit the `.env` file at the root of the project:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   INDEX_NAME=your_index_name
   ```

## Running Course Scripts

### Chapter 1: Agents and Tools
```bash
python 01_agents_and_tools/main.py
python 01_agents_and_tools/tool_calling.py
```

### Chapter 2: RAG Pipeline & Retrieval Agent
```bash
# Ingest document chunks into Pinecone
python 02_rag_implementation/rag_ingestion.py

# Query the RAG Agent
python 02_rag_implementation/rag_agent.py "What is the CGST registration threshold?"
```
