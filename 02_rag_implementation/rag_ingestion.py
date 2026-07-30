import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def validate_env() -> dict[str, str]:
    """Validate required environment variables for RAG ingestion."""
    required_vars = ["PINECONE_API_KEY", "GOOGLE_API_KEY", "INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please check your .env file."
        )

    return {var: os.environ[var] for var in required_vars}


def load_document(file_path: str) -> list[Document]:
    """Load text document from the specified file path."""
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into smaller chunks using token-based RecursiveCharacterTextSplitter.
    Preserves original metadata and attaches chunk indexing metadata.
    """
    print(
        f"Splitting document using token-based RecursiveCharacterTextSplitter "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})..."
    )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total_chunks

    print(f"Created {total_chunks} chunks.")
    return chunks


def ingest_to_vectorstore(
    chunks: list[Document],
    index_name: str,
    batch_size: int = 100,
) -> PineconeVectorStore:
    """
    Embed document chunks and ingest into Pinecone vector store in batches
    using explicit IDs and the modern PineconeVectorStore constructor pattern.
    """
    print("Initializing embeddings (GoogleGenerativeAIEmbeddings)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=1536,
    )

    print(f"Initializing PineconeVectorStore for index '{index_name}'...")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    all_ids = []
    for chunk in chunks:
        source_name = Path(chunk.metadata.get("source", "doc")).stem
        chunk_idx = chunk.metadata.get("chunk_index", 0)
        all_ids.append(f"{source_name}-chunk-{chunk_idx}")

    total_chunks = len(chunks)
    print(
        f"Ingesting {total_chunks} chunks into Pinecone index '{index_name}' in batches of {batch_size}..."
    )

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size

        print(
            f"Uploading batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)..."
        )
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)

    print("Batch ingestion complete!")
    return vectorstore


def main():
    env_vars = validate_env()

    current_dir = Path(__file__).parent
    file_path = str(current_dir / "CGST-Act-Updated-30092020.txt")
    documents = load_document(file_path)

    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    ingest_to_vectorstore(
        chunks=chunks, index_name=env_vars["INDEX_NAME"], batch_size=100
    )


if __name__ == "__main__":
    main()
