
import google.generativeai as genai
import streamlit as st

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list:
    try:
        if not text or not text.strip():
            raise ValueError("The text chunk is empty.")

        if len(text) > 12000:
            text = text[:12000]

        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )

        embedding = response.get("embedding", [])

        if not embedding:
            raise ValueError("Google returned an empty embedding.")

        return embedding

    except Exception as e:
        st.error(
            f"Google embedding failed ({task_type}): "
            f"{type(e).__name__}: {e}"
        )
        return []
