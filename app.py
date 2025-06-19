import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic
from utils.embedder import embed_text
from utils.retriever import load_index, search
import requests

def get_tenex_price():
    url = "https://api.mexc.com/api/v3/ticker/price?symbol=TENEXUSDT"
    try:
        response = requests.get(url)
        data = response.json()
        return f"Current TENEX Price: {data['price']} USDT"
    except Exception as e:
        return f"Error fetching price: {e}"

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
index, docs = load_index()

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    query_embedding = embed_text([query.question])[0]
    retrieved_docs = search(query_embedding, index, docs)

    context = "\n\n".join(retrieved_docs)
    prompt = f"""
You are TenexBot, a helpful assistant for Tenex Finance users.

Answer the following question using only the provided context. If you don’t know the answer, say “I’m not sure.”

Context:
{context}

Question: {query.question}
Answer:
"""

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return {"response": response.content}

@app.get("/")
def root():
    return {"message": "Welcome to TenexBot RAG with Claude API."}
