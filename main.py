# --------------------------------------------------
# 1. PYDANTIC
# --------------------------------------------------
from pydantic import BaseModel, Field
from typing import List
# --------------------------------------------------
# 2. ENVIRONMENT VARIABLES
# --------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------
# 3. LANGCHAIN IMPORTS
# --------------------------------------------------
from langchain.agents import create_agent
# from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# --------------------------------------------------
# 4. OLD: TavilyClient approach
# --------------------------------------------------
# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """Search the internet for information."""
#     return f"Search results for: {tavily.search(query=query)}"
# --------------------------------------------------
# 5. PYDANTIC RESPONSE SCHEMA
# --------------------------------------------------
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
class AgentResponse(BaseModel):
    """Schema for the agent's final response"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )
# --------------------------------------------------
# 6. TAVILY SEARCH
# --------------------------------------------------
from langchain_tavily import TavilySearch
search = TavilySearch(
    max_results=5
)
# --------------------------------------------------
# 7. LLM
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)
# --------------------------------------------------
# 8. TOOLS
# --------------------------------------------------
tools = [search]
# --------------------------------------------------
# 9. AGENT
# --------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)
# --------------------------------------------------
# 10. RUN THE AGENT
# --------------------------------------------------
def main():
    print("hello from langchain")
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Search for 3 job postings for data analysts "
                    "in South India with 10-15 years of experience "
                    "with technical management and not team management. "
                    "Search LinkedIn and Naukri. "
                    "Using Tableau, Power BI, data analytics."
                )
            )
        ]
    })
    print(result)
if __name__ == "__main__":
    main()
# ==================================================
# REFERENCE NOTES
# ==================================================
# 1. PYDANTIC
# BaseModel → creates a Pydantic model/object.
# Field → adds description or validation rules to a field.
# List → represents multiple values.
#
# Example:
# name: str
# age: int
# skills: List[str]
# 2. load_dotenv()
# Loads API keys and other variables from the .env file.
#
# Example:
# GOOGLE_API_KEY=xxxx
# TAVILY_API_KEY=xxxx
# 3. LANGCHAIN IMPORTS
# create_agent → creates the LangChain agent.
# HumanMessage → represents the user's message.
# ChatGoogleGenerativeAI → connects LangChain to Gemini.
# 4. OLD TAVILYCLIENT APPROACH
# TavilyClient is Tavily's Python SDK.
# We would need @tool to turn our Python function
# into a LangChain tool.
#
# TavilySearch eliminates that extra step because
# it is already a LangChain Tool.
# 5. PYDANTIC RESPONSE SCHEMA
#
# Source:
#   url: str
# Means each source must contain a URL.
## AgentRespons:
#   answer: str
#   sources: List[Source]
## This tells the agent what the FINAL response
# should look like.
## Conceptually:
#
# {
#     "answer": "...",
#     "sources": [
#         {"url": "..."},
#         {"url": "..."}
#     ]
# }
# 6. TAVILY SEARCH
# TavilySearch is already a LangChain Tool.
# max_results=5 → return up to 5 search results.
# 7. LLM
# Gemini is the brain/reasoning model used by the agent.
# 8. TOOLS
# Give the agent access to the tools it can use.
## tools = [search]
# 9. AGENT
# response_format=AgentResponse tells LangChain:
## "Return the final answer according to
#  this Pydantic structure."
# 10. RUN THE AGENT
# agent.invoke() sends the user's request to the agent.
## The agent can:
#   1. Understand the request
#   2. Decide to use Tavily
#   3. Search the web
#   4. Process the results
#   5. Return the final response
#      according to AgentResponse
