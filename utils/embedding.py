import google.generativeai as genai

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list:
    try:
        # Sanity check: keep within token limit (~12K characters)
        if len(text) > 12000:
            text = text[:12000]

        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        return response["embedding"]
    except Exception as e:
        print("❌ Embedding Error:", e)
        return []

