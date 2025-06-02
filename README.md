
# 🤖 NLP with Transformers, LangChain and Hugging Face

This project demonstrates how to integrate Natural Language Processing (NLP) models from Hugging Face with the LangChain library, enabling text generation, summarization, and question answering, with optional GPU (CUDA) acceleration.

---

## 📦 Technologies Used

- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [CUDA (NVIDIA GPU Support)](https://developer.nvidia.com/cuda-downloads)
- [PyTorch (with CUDA support)](https://pytorch.org/get-started/locally/)

---

## ⚙️ Prerequisites

- Python 3.9+
- A [Hugging Face account](https://huggingface.co/join)
- CUDA installed (optional but recommended for GPU support)

---

## 🚀 How to Run the Project

### 📥 Install Dependencies

```bash
python -m venv venv
.\env\Scripts\activate
pip install -r requirements.txt
```

To enable GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 🔐 Authenticate with Hugging Face

```bash
huggingface-cli login
```

Paste your API key and press Enter.

---

## 📊 Why Use GPU for AI?

GPUs are designed to perform thousands of mathematical operations in parallel, greatly accelerating the training and inference of AI models. CPUs are optimized for sequential tasks, making GPUs far more efficient for neural networks and NLP workloads.

---

## 📜 Example Scripts

### ✅ `main.py`

Runs a basic text summarization using the `facebook/bart-large-cnn` model.

```bash
python main.py
```

---

### ✅ `example1.py`

Generates a personalized explanation for a given topic, tailored to a specified age, using the `mistralai/Mistral-7B-Instruct-v0.1` model.

```bash
python example1.py
```

---

### ✅ `example2.py`

Performs a customizable text summarization (short/medium/long) and allows you to ask questions about the generated summary, using multiple models:

- `facebook/bart-large-cnn` for summarization
- `facebook/bart-large` for refinement
- `deepset/roberta-base-squad2` for question answering

```bash
python example2.py
```

---

## 📦 Models Used

- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- [facebook/bart-large](https://huggingface.co/facebook/bart-large)
- [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)

---

## 📄 requirements.txt

```text
transformers
langchain
langchain-huggingface
```

---

## 📌 Notes

- Models are downloaded locally in order to run.
- CUDA and NVIDIA drivers are required for GPU support.
- Scripts were developed for a Windows environment using virtualenv, but can be easily adapted.

---

## 📚 References

- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [LangChain Docs](https://python.langchain.com/docs/)
- [Using CUDA with PyTorch](https://pytorch.org/get-started/locally/)
