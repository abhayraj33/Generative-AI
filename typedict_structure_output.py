from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing_extensions import TypedDict,Annotated
import os


load_dotenv()

class Review(TypedDict):
    
    summary:Annotated[str,...,"One sentence Summary"]
    Sentiment:Annotated[str,...,"Posistive,Negative or neutral"]

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)

structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""I love the ui but the re is issue in the batatery of the mobile phone and it it a slight laging problem in it camera quality is also not satisfying""")

print(result["summary"])

print(result["Sentiment"])


print(result)