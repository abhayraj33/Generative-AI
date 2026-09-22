from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from regex import template
from torch import chain_matmul


load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="you are a senior cricket team selector give a brief review on the {player_name} \n mention all the strong and weak point of the player ",
    input_variables=["player_name"]

)
parser=StrOutputParser()

chain=template1 | model | parser

result=chain.invoke({"player_name:vaibhav suryavanshi"})

print(result)

chain.get_graph().print_ascii()
