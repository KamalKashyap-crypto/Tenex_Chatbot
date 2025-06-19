import faiss
import pickle
import numpy as np

def load_index(index_path='embeddings/faiss_index.bin', docs_path='embeddings/docs.pkl'):
    index = faiss.read_index(index_path)
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)
    return index, docs

def search(query_embedding, index, docs, top_k=3):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [docs[i] for i in indices[0]]
