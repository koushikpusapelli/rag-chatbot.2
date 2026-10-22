

import streamlit as st
from google import genai


def generate_answer(context: str, question: str) -> str:
    st.write("Debug: Entered generate_answer()")
    st.write("Debug: Context length =", len(context or ""))
    st.write("Debug: Question =", question)

    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")

        if not api_key:
            st.error("GOOGLE_API_KEY is missing from Streamlit Secrets.")
            return ""

        if not context or not context.strip():
            st.error("Retrieved context is empty.")
            return ""

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=(
                "Answer the question using the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        )

        st.write("Debug: Gemini request completed.")

        answer = response.text
        if not answer:
            st.error("Gemini returned no text. Check its response in logs.")
            st.write(response)
            return ""

        return answer.strip()

    except Exception as e:
        st.error(f"Gemini error: {type(e).__name__}: {e}")
        st.exception(e)
        return ""
