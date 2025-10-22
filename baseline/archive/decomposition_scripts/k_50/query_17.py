# Strategy for Query 17: 1997 anime films or slayer films (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve document ids for the first part of the query: "1997 anime films"
    anime_1997_ids = retrieve("1997 anime films", k_final)
    
    # Retrieve document ids for the second part of the query: "slayer films"
    slayer_ids = retrieve("slayer films", k_final)
    
    # Combine the results using set operations
    final_doc_ids = anime_1997_ids.union(slayer_ids)
    
    # Convert the final document IDs to a list for the result
    result = list(final_doc_ids)
    return result
