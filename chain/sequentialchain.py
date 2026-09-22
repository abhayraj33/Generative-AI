from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",

)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1=PromptTemplate(
    template="you are a scientist in the nasa give the name of your {no_of_rocket}rocket that perform well according to you  ",
    input_variables=["no_of_rocket"]
)

template2=PromptTemplate(
    template="generate a 7 line summary n the topic {topic}",
    input_variables=["topic"]
)


chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"no_of_rocket:5"})

print(result)