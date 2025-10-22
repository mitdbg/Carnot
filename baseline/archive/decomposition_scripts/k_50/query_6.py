# Strategy for Query 6: Arecaceae that are trees of Indo-China (No Truncation)
def execute_query(retrieve):
    k_final = 50
    # Step 1: Retrieve documents about Arecaceae
    documents_arecaceae = retrieve("Arecaceae", k_final)
    
    # Step 2: Retrieve documents about trees
    documents_trees = retrieve("trees", k_final)
    
    # Step 3: Retrieve documents about Indo-China
    documents_indochina = retrieve("Indo-China", k_final)
    
    # Step 4: Combine the results using set intersection to find common documents
    final_doc_ids = documents_arecaceae & documents_trees & documents_indochina
    
    # Step 5: Convert the final set of document IDs to a list
    result = list(final_doc_ids)
    return result
