import os
from langchain_core.runnables import RunnableLambda

# Enable Tracing
os.environ["LANGSMITH_TRACING"] = "true"

# CRITICAL FOR APAC REGION: Point to the APAC API endpoint
os.environ["LANGSMITH_ENDPOINT"] = "https://apac.api.smith.langchain.com"

# Your API Key and Project Name
os.environ["LANGSMITH_PROJECT"] = "hello-world"

# Run a test
runnable = RunnableLambda(lambda x: f"Hello, {x}!")
print("Response:", runnable.invoke("LangSmith"))