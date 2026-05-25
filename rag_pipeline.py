import chromadb
import os

def load_chroma():

    # Create folder if not exists
    os.makedirs("chroma_db", exist_ok=True)

    # Load persistent database
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # Load collection
    collection = client.get_or_create_collection(
        name="skills"
    )

    return collection