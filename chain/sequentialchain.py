from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from  langchain_google_genai import ChatGoogleGenerativeAI
import os 
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
 # we are using it  because we wanted to use two different model here we are using nvidia api key throught the openroughter

# model=ChatOpenAI(
#     model="nvidia/nemotron-3-ultra-550b-a55b:free",
#     openai_api_base="https://openrouter.ai/api/v1",
#     openai_api_key=api_key
# )

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
)
model=ChatHuggingFace(llm=llm)
model1=ChatHuggingFace(llm=llm)





# model1=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser=StrOutputParser()


text="""
Humanity is a rich concept that can be understood from several angles — biological, psychological, social, cultural, and philosophical. Here's a detailed look:

1. Biological Perspective

Humans (Homo sapiens) are a species of primates that evolved roughly 300,000 years ago in Africa. Biologically, what sets us apart includes:

A highly developed prefrontal cortex, enabling complex reasoning, planning, and self-control
Bipedalism (walking upright), which freed our hands for tool use
Opposable thumbs, allowing fine motor skills and craftsmanship
A long childhood development period, giving more time for learning and brain development compared to other species

2. Cognitive and Psychological Traits

Humans are defined by unique cognitive abilities:

Abstract thinking — ability to imagine things that don't exist yet, plan for the future, and understand symbols
Language — a sophisticated system of communication far more complex than any other species, allowing knowledge to be recorded and passed down
Self-awareness — humans can reflect on their own existence, mortality, and consciousness
Emotional depth — complex emotions like guilt, pride, nostalgia, and empathy, which shape behavior and relationships

3. Social Nature

Humans are inherently social creatures:

We form families, communities, and nations, built on cooperation, trust, and shared norms
Culture — humans transmit knowledge, values, and traditions across generations, not just genetically but socially
Morality and ethics — humans develop systems of right and wrong, fairness, and justice, which guide individual and collective behavior
Empathy and altruism — humans often help others even at personal cost, a trait that, while seen in some animals, is especially pronounced in humans

4. Creativity and Innovation

Humanity is marked by an extraordinary capacity to create:

Art, music, literature — expressions of emotion, imagination, and meaning
Science and technology — the drive to understand the natural world and build tools to shape it, from fire to spacecraft
Philosophy and religion — attempts to answer fundamental questions about existence, purpose, and meaning

5. Contradictions Within Humanity

Humanity is also defined by deep contradictions:

Capacity for immense kindness, generosity, and cooperation, alongside capacity for cruelty, violence, and destruction
Pursuit of knowledge and truth, yet also prone to bias, superstition, and self-deception
Ability to build complex civilizations, yet also capable of war and environmental destruction

6. Philosophical Views on Humanity

Different philosophical traditions describe humanity differently:

Humanism emphasizes human dignity, reason, and potential for self-improvement
Existentialism focuses on individual freedom, choice, and the search for meaning in an indifferent universe
Religious perspectives often see humans as spiritual beings with a purpose tied to the divine
Evolutionary perspectives see humans as products of natural selection, driven by survival and reproduction, yet capable of transcending pure instinct through culture and reason

In essence

Humanity is the collective story of a species that is simultaneously biological and cultural, rational and emotional, individual and communal. We are capable of extraordinary compassion and creativity, but also of great harm — and much of what makes us "human" lies in this tension, and in our ongoing struggle to understand ourselves and improve """

template1=PromptTemplate(
    template="generate  notes on the given topic {text}  ",
    input_variables=["text"]
)

template2=PromptTemplate(
    template="generate 5 question and answer on {text}",
    input_variables=["text"]
)
template3=PromptTemplate(
    template="on the basis of given topic {notes} write a summary of the notes and also here re the 5 question answer {question} make the language easy of the questions and answer ",
    input_variables=["notes","question"]
)

parell_chain=RunnableParallel({
    "notes":template1 | model | parser,
    "question":template2 | model | parser
})

merge_Chain= template3 | model | parser

# adding the chain 
chain=parell_chain | merge_Chain

result=chain.invoke({"text":text})

print(result)

# result=parell_chain.invoke({"task":text})

# print(result.content)
chain.get_graph().print_ascii()
