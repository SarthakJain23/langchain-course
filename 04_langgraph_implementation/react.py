from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: int) -> int:
    """
    Return 3 times the value of num
    """
    return num * 3


tools = [triple, TavilySearch(max_results=1)]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0).bind_tools(tools)
