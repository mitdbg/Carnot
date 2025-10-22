# create_chroma.py

# --- FIX FOR SQLITE3 VERSION (from our previous conversation) ---
# This must be at the very top of your script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------

import chromadb

# Initialize the client. This is an in-memory instance.
client = chromadb.Client()

# Get the collection object, creating it if it doesn't exist.
# This is safer and prevents errors on subsequent runs.
collection = client.get_or_create_collection(name="quest_documents")

print("Successfully connected to Chroma and got the collection!")
print(f"There are currently {collection.count()} documents in the collection.")