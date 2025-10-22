# Strategy for Query 11: American action films about security and surveillance also bullying (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve sets of document IDs based on the different parts of the query
    action_films = retrieve("American action films", k_final)
    security_surveillance = retrieve("security and surveillance", k_final)
    bullying = retrieve("bullying", k_final)
    
    # Combine the sets using set intersection and union to find relevant documents
    final_doc_ids = action_films.intersection(security_surveillance).union(bullying)
    
    # Convert the final set to a list
    result = list(final_doc_ids)
    return result
