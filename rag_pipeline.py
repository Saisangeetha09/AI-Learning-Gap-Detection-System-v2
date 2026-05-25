import chromadb
import os

def load_chroma():

    # Create folder if not exists
    os.makedirs("chroma_db", exist_ok=True)

    # Load persistent database
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # Load collection safely
    try:
        collection = client.get_collection(
            name="skills"
        )

    except:

        collection = client.create_collection(
            name="skills"
        )

    return collection