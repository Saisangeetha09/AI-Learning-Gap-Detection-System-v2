import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/skills_dataset.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Create collection
collection = client.get_or_create_collection(name="skills")

# Convert dataset rows into documents
documents = []

for index, row in df.iterrows():

    text = f"""
    Skill: {row['skill_name']}
    Category: {row['category']}
    Difficulty: {row['difficulty_level']}
    Prerequisites: {row['prerequisites']}
    Learning Time: {row['learning_time_days']} days
    """

    documents.append(text)

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store embeddings in ChromaDB
for i, doc in enumerate(documents):

    collection.add(
        documents=[doc],
        embeddings=[embeddings[i]],
        ids=[str(i)]
    )

print("Embeddings stored successfully!")

# Test retrieval
query = "Machine Learning"

query_embedding = model.encode([query]).tolist()[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nRetrieved Results:\n")

for doc in results["documents"][0]:
    print(doc)