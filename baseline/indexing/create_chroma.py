# create_chroma.py

# --- FIX FOR SQLITE3 VERSION ---
# This must be at the very top of your script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------

import chromadb
import os
# chroma_quest_limited
PERSIST_DIR = "./chroma_quest_limited_2"
COLLECTION_NAME = "quest_documents_limited_2"

# This will load the database from the directory if it exists
# or create it if it doesn't.
os.makedirs(PERSIST_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Get the collection object, creating it if it doesn't exist.
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Successfully connected to Chroma (PersistentClient at {PERSIST_DIR})")
print(f"Got collection: '{COLLECTION_NAME}'")
print(f"There are currently {collection.count()} documents in the collection.")