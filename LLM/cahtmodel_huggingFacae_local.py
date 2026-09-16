from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
          temprature=0.5
    )

)

model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of india")
print(result)


# import os
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="meta-llama/Llama-3.1-8B",
#     task="text-generation",
#     pipeline_kwargs=dict(temperature=0.5),
#     model_kwargs={"token": os.getenv("HUGGINGFACEHUB_API_TOKEN")}
# )