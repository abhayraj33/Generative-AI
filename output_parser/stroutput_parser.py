from pydoc import text
from unittest import result
from click import prompt
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from regex import template

load_dotenv()



llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

# Prompt1
template1=PromptTemplate(
    template="Write a detailed report in {topic}",
    input_variables=["topic"]
)

# Prompt2
template2=PromptTemplate(
    template="Write a 5 line summary on the text \n {text}",
    input_variables=["text"]
)

prompt1=template1.invoke({"topic:Black hole"})

result=model.invoke(prompt1)

prompt2=template2.invoke({"text":result.content})

result1=model.invoke(prompt2)


print(result1.content)