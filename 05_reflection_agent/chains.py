from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral linkedin content writer grading a linkedin post. Generate critique and recommendations for the user's post. Always provide detailed recommendations, including requests for length, virality, style etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral linkedin content writer tasked with writing excellent linkedin posts. "
            "Generate the best linkedin post possible for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
