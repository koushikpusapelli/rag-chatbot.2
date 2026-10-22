
import os
import streamlit as st
from google import genai
from google.genai import types


def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list:
    try:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from Streamlit Secrets.")

        client = genai.Client(api_key=api_key)

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text[:12000],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )

        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("Google returned an empty embedding.")

        embedding = response.embeddings[0].values

        if len(embedding) != 768:
            raise ValueError(
                f"Expected 768 dimensions, received {len(embedding)}."
            )

        return embedding

    except Exception as e:
        st.error(f"Embedding Error: {type(e).__name__}: {e}")
        return []
