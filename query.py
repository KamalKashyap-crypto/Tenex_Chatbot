import os
import pickle
from dotenv import load_dotenv
import numpy as np
import anthropic
from utils.embeddings import embed_text
from utils.vector_store import cosine_similarity
import requests

def get_tenex_price():
    url = "https://api.mexc.com/api/v3/ticker/price?symbol=TENEXUSDT"
    try:
        response = requests.get(url)
        data = response.json()
        return f"Current TENEX Price: {data['price']} USDT"
    except Exception as e:
        return f"Error fetching price: {e}"
    
# Load environment variables from .env
load_dotenv()
# Load index
with open("index.pkl", "rb") as f:
    index = pickle.load(f)

documents = index["documents"]
embeddings = np.array(index["embeddings"])  

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Anthropic API key not found. Make sure it's set in your .env file.")

client = anthropic.Anthropic(api_key=api_key)

while True:
    query = input("Ask something (or type 'exit' to quit): ")
    if "price" in query.lower():
        print(get_tenex_price())

    if query.lower() == "exit":
        break

    query_vector = embed_text(query).reshape(1, -1)  # Shape (1, 384)
    similarities = cosine_similarity(query_vector, embeddings).flatten()  # Shape (N,)

    # Get top matching document
    top_idx = np.argmax(similarities)
    top_document = documents[top_idx]

    # Send to Claude
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=[
            {"role": "user", "content": f"Context: {top_document}\n\nQuestion: {query}"}
        ]
    )
    print("Answer:", response.content[0].text)
