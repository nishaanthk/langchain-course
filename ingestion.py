
import os
from typing import List

from dotenv import load_dotenv
from google import genai

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


# --------------------------------------------------
# Load environment variables from .env
# --------------------------------------------------

load_dotenv()


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
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading document...")

    loader = UnstructuredLoader(
        file_path=r"C:\Users\JAY28850\Desktop\langchain-course\mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000
    )

    documents = loader.load()

    # --------------------------------------------------
    # Split document into chunks
    # --------------------------------------------------

    print("Splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    # --------------------------------------------------
    # Initialize Gemini embeddings
    # --------------------------------------------------

    embeddings = GeminiEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768
    )

    # --------------------------------------------------
    # Store embeddings in Pinecone
    # --------------------------------------------------

    print("Ingesting into Pinecone...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"]
    )

    print("Finished!")

