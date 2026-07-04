import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./embeddings/chroma_db"
)

collection = client.get_collection(
    "nit_documents"
)

query = input("Ask Question: ")

query_embedding = model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print(results["documents"])