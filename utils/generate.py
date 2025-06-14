import google.generativeai as genai

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
    Based on the context below, answer the question:

    Context:
    {context}

    Question:
    {question}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Answer Error:", e)
        return "❌ Sorry, I couldn't generate a response."
