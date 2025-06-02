from transformers import pipeline
# import torch

# Check if GPU is available
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# Model is from huggingface
# It will be installed in your machine in order to run
model = pipeline("summarization", model="facebook/bart-large-cnn", device="0") # device 0 is torch.cuda.get_device_name(0)
response = model("text to summarize")

print(response)