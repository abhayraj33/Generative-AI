from langchain_core.prompts import PromptTemplate

# template
template=PromptTemplate(
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
template.save("template.json")