from typing import Annotated, TypedDict

from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"
LAST = -1


class MessageGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def generation_node(state: MessageGraphState) -> MessageGraphState:
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def reflection_node(state: MessageGraphState) -> MessageGraphState:
    last_message = state["messages"][LAST]
    response = reflection_chain.invoke(
        {"messages": [HumanMessage(content=str(last_message.content))]}
    )
    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: MessageGraphState) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder = StateGraph(MessageGraphState)
builder.add_node(GENERATE, generation_node)
builder.set_entry_point(GENERATE)
builder.add_node(REFLECT, reflection_node)
builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()


if __name__ == "__main__":
    input = HumanMessage(content="""
    🚀 What if your company had an AI Business Analyst that could read messy financial models, multi-tab Excel sheets, PDFs, and Word docs—without sending sensitive corporate data to third-party vector cloud services?

I just built and open-sourced **Business Analyst RAG** 📊 — an enterprise-ready, local-first Retrieval-Augmented Generation system designed specifically for financial researchers, strategy teams, and business analysts.

---

💡 **The Big Problem with Typical Cloud-Based RAG Systems:**
Many enterprises rushing to adopt SaaS/Cloud RAG tools run into 3 major roadblocks:
1️⃣ **Data Privacy & Compliance Risk**: Internal financial models, P&L statements, and M&A docs cannot sit on third-party cloud vector stores.
2️⃣ **Crazy Cloud Infrastructure Costs**: Paying continuous monthly retainers for cloud vector DBs (Pinecone, cloud managed indexes) when local, embedded vector stores handle enterprise knowledge bases at a fraction of the cost.
3️⃣ **Inefficient Document Sync**: Re-indexing an entire knowledge base every time a document changes burns compute and API quotas.

---

🔥 **How this solution is built differently:**

⚡ **Local & Embedded Vector Storage**: Uses embedded ChromaDB and local state persistence—your vector indices, chunks, and metadata stay 100% on-prem / on your own infrastructure.
⚡ **Smart SHA-256 Incremental Ingestion**: Tracks file hashes so you only index ADDED or MODIFIED files and evict DELETED ones—zero redundant embedding API calls or full database rebuilds.
⚡ **Tabular Data Mastery**: Directly converts complex multi-sheet Excel files (.xlsx) and CSVs into structured Markdown tables before chunking, preserving semantic table structures that typical RAG parsers destroy.
⚡ **Strict Business Analyst Persona**: Prompt-engineered with LangGraph & Google Gemini (`gemini-flash` & `gemini-embeddings`) for quantitative extraction, financial trend analysis, and source citation with relevance match scores.
⚡ **Plug-and-Play Dashboard**: Built with an interactive Streamlit UI featuring real-time response streaming, customizable Top-K retrieval, and similarity threshold filters.

---

🏢 **Where companies can use this today:**
✅ **M&A Due Diligence**: Ingest hundreds of vendor contracts and balance sheets for instant cross-document risk discovery.
✅ **Quarterly Earnings & Strategy Briefings**: Query across 10-Ks, board decks, and strategy notes in seconds.
✅ **Operations & Supply Chain Audits**: Cross-reference multi-tab supplier spreadsheets and inventory logs.

---

🛠️ **Tech Stack**:
Python | Streamlit | ChromaDB | Google Gemini API & Embeddings | LangGraph / LangChain | Pandas & OpenPyXL

Check out the GitHub repo here 👉 [Insert your GitHub Repo Link]

I’d love to hear your thoughts: How is your team currently handling RAG for internal sensitive tabular data & spreadsheets? Drop your thoughts below! 👇

#AI #GenerativeAI #RAG #MachineLearning #BusinessIntelligence #DataScience #Python #LangChain #LLM #FinTech

    """)
    response = graph.invoke({"messages": [input]})
    print(response)