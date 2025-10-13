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

# Step 1: Check if collection exists
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # dimension of embedding model
            distance=Distance.COSINE
        )
    )
else:
    print(f"Collection '{collection_name}' already exists")

