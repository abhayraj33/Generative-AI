from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

class feedback(BaseModel):
    sentimet:Literal["positive","negative"]=Field(description="Give the sentiment of the feebback")

parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="make a sentiment analyse and tell wheter it is a negative or positive do not give  \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="write a message for the positive feedback  of our product or service \n{feedback} ",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="write a message for the negative feedback  of our product or service \n{feedback} ",
    input_variables=["feedback"]
)

feedback="today i went to the burgerking outlet with my mom and she was very to be there thanks to the burger king"

clasifier_chain=prompt1 | model | parser2

branch_Chain=RunnableBranch(
    (lambda x :x.sentimet=="positive", prompt2 | model | parser),
    (lambda x :x.sentimet=="negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Sentimentc ould not not found")
)

chain=clasifier_chain | branch_Chain

result=chain.invoke({"feedback":feedback})

print(result)

chain.get_graph().print_ascii()