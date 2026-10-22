
import streamlit as st
from google import genai


def generate_answer(context: str, question: str) -> str:
    if not context or not context.strip():
        return (
            "I couldn't find relevant context in Pinecone. "
            "Please check the retrieved chunks and try again."
        )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the retrieved context below.
If the answer is not in the context, clearly say so.
Do not invent facts.

Retrieved context:
{context}

User question:
{question}

Answer:
"""

    try:
        client = genai.Client(
            api_key=st.secrets["GOOGLE_API_KEY"]
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text.strip()

    except Exception as e:
        st.error(
            f"Gemini API error: {type(e).__name__}: {e}"
        )
        return "❌ Answer generation failed. Check the API error above."
