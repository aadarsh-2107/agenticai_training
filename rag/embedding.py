from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

load_dotenv()


def main():
    # Region: Load document
    loader = PyPDFLoader("rag/The Data Warehouse Toolkit - Kimball.pdf")
    documents = loader.load()
    # End Region

    # Region: Text splitter
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    characters = splitter.split_documents(documents)   # split the loaded docs
    # End Region

    # Region: Embedding and vector store
    print("Creating embeddings and storing in the vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma.from_documents(
        characters,
        embeddings,
        persist_directory=os.environ["RAG_VS_DIR"],
    )
    print("Stored in the embeddings")


if __name__ == "__main__":
    main()
