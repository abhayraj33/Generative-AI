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
    provider="auto"
)

model=ChatHuggingFace(llm=llm)


prompt=PromptTemplate(
    template="generate the feedbak for the producct \n {product} \n the probability of generation negative feedback is 0.5 and the probability for generating the feedback for the negative feeedback is 0.5",
    input_Variables=["product"]
)
prompt1=PromptTemplate(
    template="clasify wether the feedback  is positive or negative here is the feedback \n {feedback}",
    input_Variables=["feedback"]
)

parser=StrOutputParser()



chain=prompt | model | parser | prompt1 | model | parser

result=chain.invoke({"laptop"})

print(result)