from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

# Region Prompt

system_prompt ="""
Answer the questions based only on the following context:

{context}

Questions: {question}
Provide the detailed answer
"""
prompt_template =ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    format_str="\n\n".join(doc.page_content for doc in docs)
    return format_str

# Endregion Prompt

# Region Model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5.5")
llm_for_rag=ChatOpenAI(model="gpt-5.5",temperature=0)
# Endregion Model

# Region Retriever
vs=Chroma(persist_directory=os.environ["RAG_VS_DIR"],embedding_function=embedding)
retriever=vs.as_retriever(k=3)
# Endregion Retriever

print("Docs in collection:", vs._collection.count())
def main(query):
    # region LLM without RAG 
    print("LLM without RAG")
    result=llm.invoke(query)
    print(result.content)
    print("#"*50)
    # endreion without rag 

    # region LLM With RAG 
    docs = retriever.invoke(query)

    context = format_docs(docs)
    print("Context preview:", context[:300])

    message = prompt_template.format_messages(context = context, question=query)
    results = llm_for_rag.invoke(message)
    print(results.content)

if __name__=="__main__":

    main("What are late arriving dimensions")
