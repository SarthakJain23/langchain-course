# Chapter 1: Agents, Structured Outputs & Tool Calling

This chapter covers building LangChain agents with custom tool calling capabilities and structured response formatting.

## Scripts Included

- **[`main.py`](file:///Users/sarthakjain/Desktop/langchain-course/01_agents_and_tools/main.py)**: Implements an agent using Pydantic structured output models (`AgentResponse`, `Source`) paired with search and datetime tools.
- **[`tool_calling.py`](file:///Users/sarthakjain/Desktop/langchain-course/01_agents_and_tools/tool_calling.py)**: Demonstrates defining custom python functions as tools (`@tool`) and integrating them with Tavily search.

## How to Run

Ensure your `.env` file at the root directory contains the necessary API keys (`GOOGLE_API_KEY`, `TAVILY_API_KEY`).

From the project root:
```bash
python 01_agents_and_tools/main.py
python 01_agents_and_tools/tool_calling.py
```

Or from inside this directory:
```bash
cd 01_agents_and_tools
python main.py
python tool_calling.py
```
