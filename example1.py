
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
# import torch 

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=0)

llm = HuggingFacePipeline(pipeline=model)

# Create the promp template
template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand in a max length of 256 characters")

# Langchain chain that fills the llm with the template
chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

# Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)