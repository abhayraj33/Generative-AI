from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

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

# prompt1=template1.invoke({"topic:Black hole"})

# result=model.invoke(prompt1)

# prompt2=template2.invoke({"text":result.content})

# result1=model.invoke(prompt2)


# print(result1.content)



#  if we use the stroutput_parsrer then we canj avoid the long way of 

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic:Black hole"})

print(result)