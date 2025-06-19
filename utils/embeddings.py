from sentence_transformers import SentenceTransformer

# Single consistent embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

def embed_text(text):
    return model.encode(text)
