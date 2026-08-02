import os
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

REQUIRED_ENV_VARS = ["LANGCHAIN_DOC_INDEX_NAME", "GOOGLE_API_KEY", "PINECONE_API_KEY"]


def validate_env() -> None:
    """Validate that all required environment variables are configured."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


validate_env()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=1536,
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("LANGCHAIN_DOC_INDEX_NAME"),
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

model = init_chat_model(model="gemini-3.6-flash", model_provider="google-genai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieves relevant documentation for LangChain to help answer user queries."""
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
            for doc in docs
        ]
    )
    return context, docs


SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering LangChain documentation questions.
You have access to a tool that retrieves relevant documentation based on the user's query.

Guidelines:
- Use the retrieval tool to find relevant documentation before answering questions.
- Always cite the sources you use in your answers.
- If you cannot find the answer in the retrieved docs, state clearly that you do not know.
"""


agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)


def _format_raw_answer(raw_answer: Any) -> str:
    """Safely extracts text content from LLM response representations."""
    if isinstance(raw_answer, str):
        return raw_answer
    elif isinstance(raw_answer, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_answer
        )
    return str(raw_answer)


def _extract_deduplicated_docs(messages: List[BaseMessage]) -> List[Any]:
    """Extracts retrieved Document artifacts from ToolMessages and deduplicates them."""
    context_docs = []
    seen_sources = set()

    for message in messages:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            artifacts = getattr(message, "artifact", None)
            if isinstance(artifacts, list):
                for doc in artifacts:
                    source_key = getattr(doc, "metadata", {}).get(
                        "source"
                    ) or getattr(doc, "page_content", "")[:50]
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        context_docs.append(doc)

    return context_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved docs.

    Args:
        query (str): The user query to answer.

    Returns:
        Dict[str, Any]: Dictionary containing 'answer' and deduplicated 'context' docs.
    """
    messages = [{"role": "user", "content": query}]

    result = agent.invoke({"messages": messages})
    result_messages = result.get("messages", [])

    raw_answer = result_messages[-1].content if result_messages else ""
    answer = _format_raw_answer(raw_answer)
    context_docs = _extract_deduplicated_docs(result_messages)

    return {"answer": answer, "context": context_docs}


async def arun_llm(query: str) -> Dict[str, Any]:
    """Async variant of run_llm."""
    messages = [{"role": "user", "content": query}]

    result = await agent.ainvoke({"messages": messages})
    result_messages = result.get("messages", [])

    raw_answer = result_messages[-1].content if result_messages else ""
    answer = _format_raw_answer(raw_answer)
    context_docs = _extract_deduplicated_docs(result_messages)

    return {"answer": answer, "context": context_docs}
