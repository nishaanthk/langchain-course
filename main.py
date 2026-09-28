import os
from operator import itemgetter
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from google import genai


# --------------------------------------------------
# Custom LangChain wrapper for Google Gemini embeddings
# --------------------------------------------------

class GeminiEmbeddings(Embeddings):

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768
    ):
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"]
        )

        self.model = model
        self.output_dimensionality = output_dimensionality

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={
                "output_dimensionality": self.output_dimensionality
            }
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "output_dimensionality": self.output_dimensionality
            }
        )

        return response.embeddings[0].values


# --------------------------------------------------
# Models
# --------------------------------------------------

embeddings = GeminiEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


# --------------------------------------------------
# Pinecone Vector Store
# --------------------------------------------------

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# RAG Prompt
# --------------------------------------------------

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# ============================================================================
# IMPLEMENTATION 3: Batch Processing with LCEL
# ============================================================================
def create_retrieval_chain_with_batch():
    """
    Create an LCEL retrieval chain that can process multiple questions
    using batch().

    Instead of invoking the chain one question at a time:

        chain.invoke({"question": question1})
        chain.invoke({"question": question2})

    we can process multiple inputs:

        chain.batch([
            {"question": question1},
            {"question": question2}
        ])

    Useful when you have multiple independent questions to process.
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    ##we are just using the llm as it is. 
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

        # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)

    # ============================================================================
    # Option 3: Batch Processing with LCEL
    # ============================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 3: Batch Processing with LCEL")
    print("=" * 70)

    batch_chain = create_retrieval_chain_with_batch()

    questions = [
        "what is Pinecone in machine learning?",
        "what is a vector database?",
        "what are embeddings in machine learning?"
    ]

    results = batch_chain.batch(
        [
            {"question": question}
            for question in questions
        ]
    )

    print("\nBatch Results:")

    for i, result in enumerate(results, start=1):
        print(f"\nQuestion {i}: {questions[i - 1]}")
        print(f"Answer: {result}")