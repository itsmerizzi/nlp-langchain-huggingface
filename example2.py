from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# Remove verbosity erros from the console
set_verbosity_error()

# The goal here is to show how you can use different models together
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline("summarization", model="facebook/bart-large", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way: \n\n {text}")

# Langchain chain, with different models working together
summarization_chain = summary_template | summarizer | refiner

text_to_summarize = input("\nEnter text to summarize:\n")

length = input("n\Enter the length (short/medium/long): ")
length_map = {"short": 50, "medium": 150, "long": 300}
max_length = length_map.get(length.lower(), 150)

summary = summarization_chain.invoke({"text": text_to_summarize, "length": max_length})

print("\n **Generated Summary:** \n")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop): \n")
    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n **Answer:** ")
    print(qa_result["answer"])