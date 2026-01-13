from _internal.chroma_store import ChromaStore
import pdb

store = ChromaStore("quest_expanded_flat_subset", "./chroma_collections")

def get_doc_by_title(title: str):
    r = store.collection.get(where={"title": title}, include=["metadatas", "documents"], limit=1)
    if not r["ids"]:
        return None
    return {
        "id": r["ids"][0],
        "metadata": r["metadatas"][0],
        "text": r["documents"][0],
    }

pdb.set_trace()

doc = get_doc_by_title("Canyon wren")
print(len(doc["metadata"]))
