import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.embedding import get_embedding
from utils.retrieval import retrieve_context
from utils.generate import generate_answer
from pinecone import Pinecone
import google.generativeai as genai



load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

st.title("📄 Agent - RAG")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF Document", type=["pdf"])
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PDFPlumberLoader("temp.pdf")  # Cleaner text extraction
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        def is_valid_chunk(text: str) -> bool:
            return (
            len(text.strip()) > 100 and
            "renderx" not in text.lower() and
            "xsl•fo" not in text.lower() and
            "journal of" not in text.lower() and
            len(set(text.lower().split())) > 10
            )

        clean_chunks = [chunk for chunk in chunks if is_valid_chunk(chunk.page_content)]
    
        vector_data = []
        # ✅ Clear Pinecone index before uploading new vectors
        ###index.delete(delete_all=True)

        for chunk in clean_chunks:
            embedding = get_embedding(chunk.page_content, task_type="RETRIEVAL_DOCUMENT")
            if embedding and len(embedding) == 768:
                vector_data.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"text": chunk.page_content}
                })
            else:
                print("❌ Skipped chunk due to invalid embedding (empty or wrong size)")

        index.upsert(vectors=vector_data)
        st.success(f"✅ Uploaded and embedded {len(vector_data)} chunks.")

question = st.text_input("Ask a question:")
if question:
    embedding = get_embedding(question)
    context_chunks = retrieve_context(embedding)
    context = "\n\n".join(context_chunks)
    answer = generate_answer(context, question)
    st.markdown("### 📌 Answer")
    st.write(answer)