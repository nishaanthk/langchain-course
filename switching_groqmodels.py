import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load environment variables from .env file
load_dotenv()

# 2. Instantiate different Groq models
fast_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

heavy_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# ---------------------------------------------------------
# PATTERN 1: Explicit Model Switching (Task-Based Selection)
# ---------------------------------------------------------
def query_model(prompt: str, task_complexity: str = "simple") -> str:
    """
    Selects the appropriate model based on task requirements.
    """
    if task_complexity == "complex":
        selected_model = heavy_model
    else:
        selected_model = fast_model

    response = selected_model.invoke([("human", prompt)])
    return response.content


# ---------------------------------------------------------
# PATTERN 2: Automatic Model Switching (Fallback Resilience)
# ---------------------------------------------------------
# Automatically switches to heavy_model if fast_model fails or hits a rate limit (429)
fallback_model = fast_model.with_fallbacks([heavy_model])


# ---------------------------------------------------------
# Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    prompt = "Explain the difference between McLaren and Ferrari in two sentences."

    print("1. Querying Fast Model (8B):")
    print(query_model(prompt, task_complexity="simple"))

    print("\n2. Querying Heavy Model (70B):")
    print(query_model(prompt, task_complexity="complex"))

    print("\n3. Querying with Automatic Fallback:")
    response = fallback_model.invoke([("human", prompt)])
    print(response.content)