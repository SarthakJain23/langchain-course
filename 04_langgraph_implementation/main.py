from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASONING = "agent_reasoning"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][LAST]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return ACT
    return END


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASONING, run_agent_reasoning)
flow.set_entry_point(AGENT_REASONING)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASONING, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASONING)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Langchain Imaplementation")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    "What is the temperature in Gurugram? List it and then triple it."
                )
            ]
        }
    )
    print(res["messages"][LAST].content[0]["text"])
