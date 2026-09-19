from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
print("Abhay")

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model=ChatHuggingFace(llm=llm)

message=[
    SystemMessage(content="you are a helping assistent")
]
while True:
    user_input=input("You :")
    message.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    result=model.invoke(message)
    message.append(AIMessage(content=result.content))
    print(result.content)

print(message)    
