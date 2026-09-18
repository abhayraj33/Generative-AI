from langchain_google_genai import ChatGoogleGenerativeAI
from  dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header("Research Paper")

user_input=st.text_input("Enter Your Pompt")

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=1.5)
if st.button("Sumirize"):
    result=model.invoke(user_input)
    st.write(result)