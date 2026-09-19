from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os
load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)
# while True:
#     user_input=input("You :")
#     if user_input=="exit":
#         break

#     result=model.invoke(user_input)
#     print("AI : ",result.content)

# This cahtbot is not saving the history os the conversation 

chat_history=[
    SystemMessage(content="You are Ai assistent Build to guide people for they wanted in a helping nature and in a kind manner ")
]
while True:
    user_input=input("You :")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)
print(chat_history)    