import pickle
from utils.embeddings import embed_text

documents = [
    "Tenex is a decentralized finance platform for token swaps, staking, and yield farming.",
    "Users can trade crypto assets securely on Tenex with low fees.",
    "Tenex offers innovative features like liquidity mining and governance voting."
]

embeddings = [embed_text(doc) for doc in documents]

index = {
    "documents": documents,
    "embeddings": embeddings
}

with open("index.pkl", "wb") as f:
    pickle.dump(index, f)

print("Index generated and saved to index.pkl")
