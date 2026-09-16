nv import load_dotenv
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
print(result