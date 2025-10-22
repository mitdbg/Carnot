# Strategy for Query 14: find me Epistemology books (No Truncation)
def execute_query(retrieve):
    k_final = 50
    epistemology_docs = retrieve("Epistemology books", k_final)
    philosophy_docs = retrieve("Philosophy books", k_final)
    final_doc_ids = epistemology_docs.union(philosophy_docs)
    result = list(final_doc_ids)
    return result
