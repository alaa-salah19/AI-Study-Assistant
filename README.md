# 🤖 AI Study Assistant: RAG-Powered PDF Chat & Quiz Generator

An advanced, end-to-end AI application built with **Python**, **LangChain**, and **Streamlit**. This tool allows users to upload PDF documents and interact with them using **Llama 3.1** via Hugging Face Inference API.

---

## 🌟 Key Features

* **💬 RAG-Based Chat:** Ask complex questions about your PDF and get context-aware answers using **FAISS** vector storage.
* **📝 Automated Exam Simulator:** Generates Multiple Choice Questions (MCQs) with instant feedback and detailed explanations using **Llama 3.1**.
* **🗒️ Smart Summarizer:** A multi-stage recursive summarization engine that handles large documents by chunking text effectively.
* **🚀 Real-time UI:** Built with **Streamlit** for a smooth, interactive, and responsive user experience.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **LLM Orchestration:** LangChain
* **Model:** Meta-Llama/Llama-3.1-8B-Instruct (via Hugging Face API)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit

---


## 💡 How It Works (RAG Workflow)
The application follows the **Retrieval-Augmented Generation** architecture:
1. **Document Loading:** PDF text is extracted and cleaned.
2. **Chunking:** Text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding:** Chunks are converted into numerical vectors.
4. **Vector Store:** Vectors are stored in **FAISS** for fast similarity search.
5. **Generation:** When a user asks a question, the system retrieves relevant chunks and sends them to **Llama 3.1** to generate a grounded response.

