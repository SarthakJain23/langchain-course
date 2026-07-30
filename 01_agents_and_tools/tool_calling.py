from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'"""
    return x * y


def main():
    tools = [TavilySearch(), multiply]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant",
    )

    response = agent.invoke({"messages": [("user", "What is 12 multiplied by 15?")]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
