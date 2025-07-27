
# 🧠 Agent-RAG: PDF-Chatbot with Gemini + Pinecone

🌐 **Live Demo**: https://rag-chatbot2-bykoushik.streamlit.app/  


## 📄 What is Agent-RAG?

Agent-RAG is an intelligent PDF-based chatbot that allows users to upload **one or more PDF files**, ask questions about them, and get accurate answers using **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini embeddings** and **Pinecone vector database**.

---

## ⚙️ What It Does

- ✅ Upload **multiple PDF files** at once  
- ✅ Each file is split into **smart chunks** using LangChain's `RecursiveCharacterTextSplitter`  
- ✅ Chunks are embedded using **Gemini's embedding model**
- ✅ Embeddings are stored in **Pinecone**
- ✅ User can query by:
  - All documents
  - Specific document (via dropdown)
- ✅ Answers are generated using **Google Gemini Pro** with context retrieved from Pinecone

---

## 🔁 How the RAG Pipeline Works

1. **Upload PDFs** ➝ Processed and split into chunks
2. **Embed Chunks** ➝ Gemini creates 768-dimensional vectors
3. **Store in Pinecone** ➝ Vectors stored along with metadata (`text`, `source`)
4. **Ask Question** ➝ Gemini embeds question ➝ Pinecone retrieves top-K similar chunks
5. **Generate Answer** ➝ Gemini generates a natural answer using retrieved context

---

## 🛠️ Tools & Frameworks Used

| Component         | Tool / Library                      |
|------------------|--------------------------------------|
| Embeddings        | 🔹 Google Gemini (`google-generativeai`) |
| Vector Store      | 🔹 Pinecone                          |
| PDF Parsing       | 🔹 `pdfplumber` via LangChain        |
| Chunking          | 🔹 `RecursiveCharacterTextSplitter`  |
| LLM               | 🔹 Gemini Pro                        |
| Frontend (MVP)    | 🔹 Streamlit                         |
| Backend Utilities | 🔹 FastAPI-ready Python modules      |
| Environment       | 🔹 Python 3.10+, `.env` configs      |

---

## 📦 Installation & Running Locally

```bash
git clone https://github.com/koushikpusapelli/rag-chatbot.2.git
cd rag-chatbot.2

# Set up virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# OR
source .venv/bin/activate      # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file with your keys
touch .env

