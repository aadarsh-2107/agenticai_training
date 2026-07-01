from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
model=ChatOpenAI(model_name="gpt5.5", temperature=0)

prompt="""
write a essay on the history of ANN (Artificial Neurla Network) 100 words
"""
if __name__=="__main__":
    print("Hello Ai Agent")
    response = model.invoke((prompt))
    print(response.content)