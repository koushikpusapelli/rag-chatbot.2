from fastapi import FastAPI
from pydantic import BaseModel
from utils.embedding import get_embedding
from utils.retrieval import retrieve_context
from utils.generate import generate_answer
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    question = query.question
    embedding = get_embedding(question)
    context_chunks = retrieve_context(embedding)
    context_text = "\n\n".join(context_chunks)
    answer = generate_answer(context_text, question)
    return {"answer": answer}
