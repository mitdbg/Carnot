# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve documents for 1912 films
    films_1912 = retrieve("1912 films", k_final)
    
    # Retrieve documents for films set in England
    films_set_in_england = retrieve("films set in England", k_final)
    
    # Combine results using set intersection
    final_doc_ids = films_1912 & films_set_in_england
    
    result = list(final_doc_ids)
    return result
