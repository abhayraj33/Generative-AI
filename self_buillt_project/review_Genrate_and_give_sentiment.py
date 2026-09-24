from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
)

model=ChatHuggingFace(llm=llm)

# chat_history=[
#     SystemMessage(content="you are a ai agent that is build for generating the feeedback for the product generate both positive and negative feedback ")

# ]

prompt=PromptTemplate(
    template="write a feedback for the product \n {product} \n do not write both at once if you are writing a positive then write only positive and if you are writing negative then write onlhy a negative feedback bsdk kabhi negative feedback bhi de de hmesa bss positive hi de rha hai tu thoda s bahan ka loda hai kya kabhi negative de kabhi positive de bss consistently ek feedback hi kyo  de rha hai ",
    input_Variables=["product"]
)

parser=StrOutputParser()



chain=prompt | model | parser

result=chain.invoke({"laptop"})

print(result)