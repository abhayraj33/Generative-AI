from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=0,max_output_tokens=10)

result = llm.invoke("suggest me 10 asthetic and unique indian girlfriend name ")

print(result.content[0]['text'])
