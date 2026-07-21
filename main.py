from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


tavily_search = TavilySearch()


@tool
def get_current_date() -> str:
    """
    Tool that gets the current date
    Returns:
        string which is the current date
    """
    return datetime.now().strftime("%Y-%m-%d")


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
tools = [tavily_search, get_current_date]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for job openings in gurgaon for Full Stack Javascript and typescript engineer on linkedin."
            )
        }
    )
    print(response)


if __name__ == "__main__":
    main()
