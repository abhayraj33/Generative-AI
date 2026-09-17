from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

st.header("Research Paper")
paper_input=st.selectbox("Select research Paper",["Attention is all you need","Bert:Pre-training of deep bidirectional Transformer","GPT-3:Lnaguage models are few sort learner","Diffusion models beats gans on Image synthasis"])
style_input=st.selectbox("select the style of the research",["Begineer friendly","Technical","code oriented","mathematical"])

input_lenth=st.selectbox("select  the length of the out put",["short (1-2 patagraph)","mediam(5-10 paragraph)","Long (datail explantion)"])

# template
template=PromptTamplate(
    template="""
    please summrize the research paper titled "{paper_input}" with the following specification:
    Explation style:"{style_input}"
    Explation length:"{input_length}"
    1.Mathematical detailes:
    include relevent formulas is present in the paper,
    Exppain the mathematical concepts using the simple, intuitive code snipet where applicable.

    2.Analogoes:
    use relates analogies to simplify complex ideas
    If certain information is not available in the paper ,respond with "Insufficient information available" insted of gussing
    Ensure the summary is clear and accurate
    """,
    input_variable=["paper_input","style_input","input_length"]


)

# fill the placeholder
prompt=template.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "input_length":input_length


})


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=0)


if st.button("Sumirize"):
    result=model.invoke(prompt)
    st.text(result.content)










