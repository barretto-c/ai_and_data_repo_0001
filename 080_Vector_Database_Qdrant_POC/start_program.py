from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Start Qdrant fully in-memory (no server)
client = QdrantClient(":memory:")

# 2. Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Sample country descriptions
countries = [
    {"id": 1, "name": "Japan", "desc": "An island nation in East Asia known for its technology, cherry blossoms, and Mount Fuji."},
    {"id": 2, "name": "Brazil", "desc": "The largest country in South America, famous for the Amazon rainforest and Carnival festival."},
    {"id": 3, "name": "Egypt", "desc": "A country in North Africa, home to ancient pyramids and the Nile River."},
    {"id": 4, "name": "Canada", "desc": "A large country in North America known for its natural beauty and multicultural cities."},
    {"id": 5, "name": "Australia", "desc": "A country and continent surrounded by the Indian and Pacific oceans, famous for the Outback and Great Barrier Reef."},
]

# 4. Create a vector collection
# Define collection with 384-dimensional vectors and cosine distance metric
# Cosine tells you how similar their directions are.
# If the arrows point the same way, cosine = 1
# If they point at a right angle, cosine = 0
# If they point in opposite directions, cosine = –1

client.recreate_collection(
    collection_name="countries",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 5. Insert vectors
vectors = [model.encode(c["desc"]).tolist() for c in countries]

client.upsert(
    collection_name="countries",
    points=[
        PointStruct(
            id=c["id"],
            vector=v,
            payload={"name": c["name"], "desc": c["desc"]}
        )
        for c, v in zip(countries, vectors)
    ]
)

# 6. Query for similar countries
query = "A country with ancient pyramids and a famous river"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="countries",
    query=query_vector,
    limit=3
)

print("\nSearch results:")
for r in results.points:
    print(r.payload, "score:", r.score)
