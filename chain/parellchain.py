from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm1=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

llm2=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm1)

model2=ChatHuggingFace(llm=llm2)

parser=StrOutputParser()

template1=PromptTemplate(
    template="You are a ai agent and you are very intelligent behave like a IIT professor and give a brief detail on the given topic{topic}",
    input_variable=["topic"]
)

template2=PromptTemplate(
    template=" generate 5 quize question on the basis of given text  {text}",
    input_Variable=["text"]
)


template3=PromptTemplate(
    template="here is the  details of the topic{topic} and also 5 quiz question combine  both of them and give me both detailed description and 5 quize quez=stion "
)

chain= template1 | model1 | parser |template2 | model2 | parser | template3 | model1 |parser

result=chain.invoke({"generative_AI"})

print(result)