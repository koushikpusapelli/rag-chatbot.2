import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google import genai as new_genai
import streamlit as st
import uuid
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embedding import get_embedding
from utils.retrieval import retrieve_context
from utils.generate import generate_answer
from pinecone import Pinecone
import google.generativeai as genai


# --- Environment Setup ---
load_dotenv()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])

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

# --- Run RAG Pipeline ---
if question:
    try:
        with st.spinner("Searching PDF and generating answer..."):
            query_embedding = get_embedding(
                question,
                task_type="RETRIEVAL_QUERY"
            )

            if not query_embedding:
                st.error("Question embedding failed.")
                st.stop()

            context_chunks = retrieve_context(
                query_embedding,
                selected_file=selected_file,
                top_k=top_k
            )

            st.markdown("### 🔍 Retrieved Context")
            st.write(f"Retrieved {len(context_chunks)} chunks.")

            if not context_chunks:
                st.warning(
                    "No relevant chunks were found in Pinecone. "
                    "Try selecting All Files or uploading the PDF again."
                )
                st.stop()

            context = "\n\n".join(context_chunks)

            # Direct Gemini call: bypass utils.generate.py
            client = new_genai.Client(
                api_key=st.secrets["GOOGLE_API_KEY"]
            )

            prompt = f"""
Answer the question using the PDF context below.
If the answer is not in the context, say so.

PDF context:
{context}

Question:
{question}
"""

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            st.markdown("### 📌 Answer")

            if response.text:
                st.write(response.text)
            else:
                st.warning("Gemini returned an empty response.")
                st.write(response)

            with st.expander("🔍 View Retrieved Context Chunks"):
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)

    except Exception as e:
        st.error(f"Pipeline error: {type(e).__name__}: {e}")
        st.exception(e)

