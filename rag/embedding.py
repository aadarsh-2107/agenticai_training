from dotenv import load_dotenv
from langchain_community.document_loaders import pyPDFLoader
from chain_lang_splitter import splitter
import os
load_dotenv()

def main():
    # Region Document header
    loader = pyPDFLoader("rag/The Data Warehouse Toolkit - Kimball.pdf")

    # END REGION
    # REGION TEXT SPLITTER
    test = CharacterTextSplitter(chunk_size, chunk_overlap = 0)
    characters = splitter.split_documents(test)
    #  End Region

    # Region Embedding and Vector Store
    print("Print Embedding into the vector")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma.from_documents(characters, embeddings, persist_directory=os.environ["RAG_VS_DIR"])
    print("Stored in the Embeddings")

if __name__=="__main__":
    main()