import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import uuid
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.embedding import get_embedding
from utils.retrieval import retrieve_context
from utils.generate import generate_answer
from pinecone import Pinecone
import google.generativeai as genai

# --- Environment Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# --- UI Title ---
st.title("🧠 Agent-RAG: Smart PDF Question Answering")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    uploaded_files = st.file_uploader(
        "📂 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

# --- Document Loading and Chunking ---
all_chunks = []

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)...")

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PDFPlumberLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        all_chunks.extend(chunks)

    st.success(f"✅ Created {len(all_chunks)} chunks.")

    # --- Embedding and Upsert ---
    vector_data = []
    for chunk in all_chunks:
        try:
            embedding = get_embedding(chunk.page_content, task_type="RETRIEVAL_DOCUMENT")
            if embedding and len(embedding) == 768:
                vector_data.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown")
                    }
                })
        except Exception as e:
            st.error(f"❌ Embedding Error: {e}")

    if vector_data:
        index.upsert(vectors=vector_data)
        st.success(f"🎉 Uploaded {len(vector_data)} vectors to Pinecone.")
    else:
        st.error("❌ No valid chunks were embedded.")

# --- Document Selector ---
file_options = ["All Files"] + list(set([chunk.metadata.get("source", "unknown") for chunk in all_chunks]))
selected_file = st.selectbox("📄 Choose document to query:", options=file_options)

# --- Top K Chunks Slider ---
top_k = st.slider("🔢 Number of chunks to retrieve", 1, 10, 3)

# --- User Question ---
question = st.text_input("💬 Ask a question:")

# --- Run RAG Pipeline ---
if question:
    query_embedding = get_embedding(question, task_type="RETRIEVAL_QUERY")
    context_chunks = retrieve_context(query_embedding, selected_file=selected_file, top_k=top_k)
    context = "\n\n".join(context_chunks)
    answer = generate_answer(context, question)

    # --- Display Output ---
    st.markdown("### 📌 Answer")
    st.write(answer)

    with st.expander("🔍 View Retrieved Context Chunks"):
        for i, chunk in enumerate(context_chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)
