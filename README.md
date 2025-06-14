#Gemini-Powered PDF Chatbot (RAG-Based)

🔗 **Live Demo**: [Click here to try the chatbot](https://rag-chatbot2-bykoushik.streamlit.app/)  


---

# What is this chatbot about?

This chatbot allows users to **upload PDF documents** (like research papers, reports, etc.), and then **ask questions** about the content. It uses the **RAG (Retrieval-Augmented Generation)** pipeline to generate highly accurate, context-aware answers using Google's Gemini model.

---

## 💡 What does it do?

- Lets you upload and analyze any PDF
- Splits documents into small text chunks
- Embeds these chunks using Gemini's embedding model
- Stores them in Pinecone (vector DB)
- Retrieves relevant chunks when a question is asked
- Uses Gemini Pro to generate answers using the retrieved context

---

## How does the RAG pipeline work?

1. **PDF Upload**: You upload a document via the Streamlit UI.
2. **Chunking**: The document is split into smaller chunks using LangChain.
3. **Embedding**: Each chunk is converted to a numerical vector using Gemini's embedding model (`embedding-001`).
4. **Storage**: These vectors are stored in Pinecone, a fast and scalable vector database.
5. **Retrieval**: When you ask a question, the most relevant chunks are retrieved from Pinecone based on similarity.
6. **Generation**: Gemini Pro uses the retrieved context to answer your question intelligently.

---

## 🛠️ Tech Stack Used

- **Streamlit**: User interface (UI)
- **Google Generative AI (Gemini)**:
  - `embedding-001` model for vector generation
  - `gemini-pro` for answer generation
- **Pinecone**: Vector database for similarity search
- **LangChain**: Chunking & PDF handling
- **Python 3.11**

---

##  Future Development Ideas

- 📁 **Upload entire folders** and recursively process all documents inside
- ☁️ **Google Drive integration**: Embed and query documents directly from Google Drive
- 🌐 **Web-enabled RAG**: Add ability to search the internet and fetch live information
- 🧠 **Document memory**: Track and summarize previously asked questions
- 👥 **Multi-user support**: Let multiple users manage their own private document knowledge base

---

## 🙌 Built With

Made using open-source tools and APIs to explore how RAG + Gemini can be used for powerful document understanding and question answering.

---

