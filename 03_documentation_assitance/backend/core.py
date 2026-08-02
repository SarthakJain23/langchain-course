from langchain_core.messages import ToolMessage
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=1536,
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("LANGCHAIN_DOC_INDEX_NAME"), embedding=embeddings
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
    """Retrieves relevant documentation for langchain to help user queries about langchain."""

    docs = retriever.invoke(query)
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
            for doc in docs
        ]
    )
    return context, docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved docs.

    Args:
        query (str): The query to answer.

    Returns:
        Dictionary containing:
            - answer : The answer to the query.
            - context : List of retrieved docs
    """

    system_prompt = (
        "You are a helpful AI assistant specialized in answering LangChain documentation questions. "
        "You have access to a tool that retrieves relevant documentation based on the user's query."
        "Use the tool to find relevant information before answering questions"
        "Always cite the sources you use in your answers"
        "If you cannot find the answer in the retrieved docs, say so"
    )

    agent = create_agent(
        model=model, tools=[retrieve_context], system_prompt=system_prompt
    )
    messages = [{"role": "user", "content": query}]
    result = agent.invoke({"messages": messages})
    answer = result.get("messages")[-1].content
    context_docs = []

    for message in result.get("messages"):
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}

