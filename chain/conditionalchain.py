from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="make a sentiment analyse and tell wheter it is a negative or positive do not give anythin insted of negative or positive i wanted to reveive only either negative or positive nothing else text as a output \n {feedback}",
    input_variables=["feedback"]
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

chain=prompt1 | model | parser

result=chain.invoke({"feedback":feedback})

print(result)


