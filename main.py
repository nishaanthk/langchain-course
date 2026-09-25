from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
# from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# --------------------------------------------------
# OLD: TavilyClient approach
# --------------------------------------------------
# from tavily import TavilyClient
# tavily = TavilyClient()
# This invokes the Tavily API using the TAVILY_API_KEY
# from your environment variables.
# @tool
# def search(query: str) -> str:
#     """Search the internet for information.
##     Args:
#         query: The search query string.
##     Returns:
#         Search results.
#     """
#     return f"Search results for: {tavily.search(query=query)}"
# --------------------------------------------------
# NEW: LangChain TavilySearch approach
# --------------------------------------------------

from langchain_tavily import TavilySearch
search = TavilySearch(
    max_results=5
)
# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)
# TavilySearch is already a LangChain Tool,
# so we can pass it directly to the agent.
tools = [search]
# --------------------------------------------------
# Agent
# --------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools
)
# --------------------------------------------------
# Run the agent
# -------------------------------------------------
def main():
    print("hello from langchain")
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content="What's the weather in Tokyo?"
            )
        ]
    })
    print(result)
if __name__ == "__main__":
    main()