from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st

load_dotenv()

st.header("Research Paper")
paper_input=st.selectbox("Select research Paper",["Attention is all you need","Bert:Pre-training of deep bidirectional Transformer","GPT-3:Lnaguage models are few sort learner","Diffusion models beats gans on Image synthasis"])
style_input=st.selectbox("select the style of the research",["Begineer friendly","Technical","code oriented","mathematical"])

input_length=st.selectbox("select  the length of the out put",["short (1-2 patagraph)","mediam(5-10 paragraph)","Long (datail explantion)"])

# template
template=load_prompt("template.json")


# fill the placeholder
# prompt=template.invoke({
#     "paper_input":paper_input,
#     "style_input":style_input,
#     "input_length":input_length


# })


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=0)


# if st.button("Sumirize"):
#     result=model.invoke(prompt)
#     st.text(result.content)

# In all this code we have two invoke 2 times which we can do in a single time using the chains


if st.button("Sumirize"):
    chain=template | model

    result=chain.invoke({
        "paper_input":paper_input,
        "style_input":style_input,
        "input_length":input_length}
    )


    st.text(result.content)


