from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Local
client = QdrantClient("localhost", port=6333)

# OR Cloud
# client = QdrantClient(
#     url="https://YOUR-CLUSTER.qdrant.io",
#     api_key="YOUR_API_KEY"
# )

collection_name = "documents"

# # Step 1: Check if collection exists
# if not client.collection_exists(collection_name=collection_name):
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(
#             size=384,  # dimension of embedding model
#             distance=Distance.COSINE
#         )
#     )
# else:
#     print(f"Collection '{collection_name}' already exists")



# # Inserting Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# docs = [
#     "Machine learning is great for predictions.",
#     "Graphs help visualize document relationships.",
#     "LLMs can summarize complex documents.",
#       "How do graphs help in document summarization?"
# ]

# embeddings = model.encode(docs)

# # Insert into Qdrant
# client.upsert(
#     collection_name="documents",
#     points=[
#         {
#             "id": idx,
#             "vector": emb.tolist(),
#             "payload": {"text": docs[idx]}  # metadata
#         }
#         for idx, emb in enumerate(embeddings)
#     ]
# )

# # Query with Similarity Search
query = "How do graphs help in document summarization?"
print("Query: ",query)
query_emb = model.encode([query])[0]

results = client.search(
    collection_name="documents",
    query_vector=query_emb.tolist(),
    limit=4
)

for r in results:
    print(r.payload["text"], r.score)
