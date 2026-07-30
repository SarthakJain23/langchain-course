import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def validate_env() -> dict[str, str]:
    """Validate required environment variables for LCEL RAG execution."""
    required_vars = ["PINECONE_API_KEY", "GOOGLE_API_KEY", "INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please check your .env file."
        )

    return {var: os.environ[var] for var in required_vars}


env_vars = validate_env()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=1536,
)

vectorstore = PineconeVectorStore(
    index_name=env_vars["INDEX_NAME"],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved document chunks with metadata headers for the prompt context."""
    if not docs:
        return "No relevant context found in the vector database."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        chunk_idx = doc.metadata.get("chunk_index", "N/A")
        formatted.append(
            f"--- Context Result {i} (Source: {source}, Chunk: {chunk_idx}) ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI assistant specializing in legal text analysis.\n"
            "Answer the user query strictly based on the following retrieved context. "
            "If the context does not contain enough information, clearly state what is missing.\n\n"
            "Retrieved Context:\n{context}",
        ),
        ("user", "{question}"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

rag_lcel_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def query_lcel_rag(question: str) -> str:
    """Execute the LCEL RAG chain for a user query."""
    print(f"\nUser Question: '{question}'")
    print("=" * 60)

    answer = rag_lcel_chain.invoke(question)

    print(f"LCEL RAG Answer:\n{answer}")
    print("=" * 60)
    return answer


def main():
    default_query = "What is the penalty for late tax filing under CGST?"

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        query_lcel_rag(query)
    else:
        print("Executing demonstration LCEL query...")
        query_lcel_rag(default_query)


if __name__ == "__main__":
    main()
