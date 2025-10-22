# Strategy for Query 9: romance films from New Zealand (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve romance films
    romance_films = retrieve("romance films", k_final)
    
    # Retrieve films from New Zealand
    nz_films = retrieve("films from New Zealand", k_final)
    
    # Combine the results using intersection to find romance films specifically from New Zealand
    final_doc_ids = romance_films.intersection(nz_films)
    
    result = list(final_doc_ids)
    return result
