
import streamlit as st
from google import genai
from google.genai import types


def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list:
    try:
        if not text or not text.strip():
            return []

        client = genai.Client(
            api_key=st.secrets["GOOGLE_API_KEY"]
        )

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text[:12000],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )

        if not response.embeddings:
            return []

        return response.embeddings[0].values or []

    except Exception as e:
        st.error(f"Embedding Error: {type(e).__name__}: {e}")
        return []
