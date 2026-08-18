from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import  ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from reflexion_chains import first_responder, revisor
from tool_executer import execute_tools

load_dotenv()


MAX_ITERATIONS = 2


def draft_node(state: MessagesState):
    """Draft the initial response"""
    res = first_responder.invoke(input={"messages": state["messages"]})
    return {"messages": [res]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results"""
    res = revisor.invoke(input={"messages": state["messages"]})
    return {"messages": [res]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Event loop that decides to continue or exit the cycle"""
    count_tool_visits = sum(isinstance(msg, ToolMessage) for msg in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tools", execute_tools)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise", event_loop, {"execute_tools": "execute_tools", END: END}
)
graph = builder.compile()


res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about Skills used in agents and how it internally works and implemented using langchain, langgraph?",
            }
        ]
    }
)