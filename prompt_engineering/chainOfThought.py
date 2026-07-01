from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI(model_name="gpt-5.5", temperature=0)

# Postfix Operation
# $ as + and ! as the *

prompt0 = """
what is the value of 5 6 ! 8 $
"""
response = model.invoke(prompt0)
print(response.content)
prompt1= """
the value of 3 4 $ is 3 + 4 = 7
then  7 7 ! is 7 * 7 = 49
Now tell me the value of 5 5 ! 9 $
 Also write "This is the new response:" before the reponse and then describe the solution
"""
response = model.invoke(prompt1)
print(response.content)