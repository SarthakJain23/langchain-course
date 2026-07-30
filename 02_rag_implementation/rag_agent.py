import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def validate_env() -> dict[str, str]:
    """Validate required environment variables for RAG retrieval."""
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

MIN_SCORE_THRESHOLD = 0.35


@tool
def retrieve_context(query: str) -> str:
    """
    Search and retrieve relevant document context from the Pinecone vector database using
    Maximal Marginal Relevance (MMR) for query diversity and score thresholding.
    Use this tool whenever you need information about the CGST Act, legal rules, or ingested documents.

    Args:
        query: Search query string to match relevant context chunks.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found in the vector database."

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    score_map = {doc.page_content: score for doc, score in docs_with_scores}

    formatted_results = []
    for i, doc in enumerate(docs, 1):
        score = score_map.get(doc.page_content)
        if score is not None and score < MIN_SCORE_THRESHOLD:
            continue

        source = doc.metadata.get("source", "Unknown Document")
        chunk_idx = doc.metadata.get("chunk_index", "N/A")
        score_str = f", Score: {score:.3f}" if score is not None else ""

        formatted_results.append(
            f"--- Context Result {i} (Source: {source}, Chunk: {chunk_idx}{score_str}) ---\n"
            f"{doc.page_content}"
        )

    if not formatted_results:
        return (
            f"Context was retrieved but none met the minimum relevance threshold "
            f"({MIN_SCORE_THRESHOLD})."
        )

    return "\n\n".join(formatted_results)


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

system_prompt = (
    "You are an expert AI assistant specializing in answering questions using Retrieval-Augmented Generation (RAG).\n"
    "When presented with a user query, ALWAYS use the `retrieve_context` tool to retrieve factual context before answering.\n"
    "Formulate your response strictly based on the retrieved context. If the context does not contain enough information, "
    "state clearly what is available and what is missing. Attribute key facts to their source document/chunk where applicable."
)

agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
)


def format_message_content(content) -> str:
    """Extract clean string text from message content, handling both strings and lists of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text"))
        return "\n".join(parts)
    return str(content)


def query_rag_agent(user_query: str) -> str:
    """
    Query the RAG agent with a user prompt and return the generated answer.

    Args:
        user_query: The natural language prompt from the user.

    Returns:
        The agent's grounded answer string.
    """
    print(f"\nUser Query: '{user_query}'")
    print("=" * 60)
    response = agent.invoke({"messages": [("user", user_query)]})
    raw_content = response["messages"][-1].content
    final_answer = format_message_content(raw_content)
    print(f"Agent Answer:\n{final_answer}")
    print("=" * 60)
    return final_answer


def main():
    """Run default query and enter interactive loop if requested."""
    default_query = "What are the key rules regarding registration under the CGST Act?"

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        query_rag_agent(query)
    else:
        print("Running demonstration query...")
        query_rag_agent(default_query)


if __name__ == "__main__":
    main()
