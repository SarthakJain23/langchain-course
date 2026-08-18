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
    What happens when you feed Swiggy's FY25 and FY26 financial reports into a custom-built, local AI Business Analyst?

    Instead of spending hours manually cross-referencing multi-page PDFs, balance sheets, and notes to accounts, I ran our **Business Analyst RAG** on both annual statements. 

    Here are two complex questions we tested (and you can see in the demo clip below 👇):

    1️⃣ *"Break down the major operational expense heads (delivery/logistics costs, employee benefits, advertising & promotional spend) as a percentage of total revenue for both FY25 and FY26. Where did Swiggy achieve operating leverage?"*
    2️⃣ *"How did the Gross Order Value (GOV) and contribution margins of Quick Commerce (Instamart) perform in FY26 compared to FY25? Is Quick Commerce inching closer to profitability?"*

    The system synthesized cross-year comparisons, extracted exact line items, calculated margin shifts, and cited every single page source with cosine similarity scores in seconds. ⏱️

    ---

    💡 **Why build a Local-First Business Analyst RAG instead of standard cloud tools?**

    For corporate strategy, finance, and M&A teams, typical cloud-based RAG presents real challenges:
    🔒 **Data Privacy & Compliance**: Sensitive balance sheets, internal forecasts, and P&L drafts never leave your local infrastructure.
    💰 **Zero Bloated Cloud Vector Bills**: Uses embedded ChromaDB and local disk persistence—no recurring cloud vector database retainers.
    ⚡ **Smart SHA-256 Incremental Ingestion**: Tracks file hashes so updating or re-syncing documents only re-indexes modified/new files, saving compute and API quotas.
    📊 **Native Tabular & Document Parsing**: Preserves complex multi-tab spreadsheets, CSVs, Word docs, and financial PDFs cleanly as structured markdown.
    🧠 **Tailored Business Analyst Persona**: Powered by LangGraph and Google Gemini (`gemini-flash` & `gemini-embeddings`) with rigorous quantitative prompt constraints and strict evidence citation.

    ---

    🏢 **Where teams can use this today:**
    ✅ **M&A Due Diligence & Deal Sourcing**: Ingest hundreds of balance sheets and contracts for automated risk checks.
    ✅ **Earnings & Competitor Intelligence**: Instantly compare multi-year annual reports across competitors.
    ✅ **Internal FP&A and Operations**: Cross-analyze supplier cost sheets and operational expense models.

    ---

    🛠️ **Tech Stack**:
    Python | Streamlit | ChromaDB | Google Gemini API & Embeddings | LangGraph / LangChain | Pandas & OpenPyXL

    Check out the full open-source project and code on GitHub 👉 [Insert your GitHub Repo Link]

    How is your team handling automated financial analysis and internal document intelligence today?

    #AI #GenerativeAI #RAG #FinancialAnalysis #Swiggy #BusinessIntelligence #Python #MachineLearning #LangChain #DataScience #LLM #FinTech

    """)
    response = graph.invoke({"messages": [input]})
    print(response["messages"][LAST].content[0]["text"])
