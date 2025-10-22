# Strategy for Query 13: Films set in Pittsburgh that are LGBT-related romance (No Truncation)
def execute_query(retrieve):
    k_final = 50
    pittsburgh_films = retrieve("Films set in Pittsburgh", k_final)
    lgbt_romance_films = retrieve("LGBT-related romance films", k_final)
    
    # Combine the two sets to find the intersection
    final_doc_ids = pittsburgh_films.intersection(lgbt_romance_films)
    result = list(final_doc_ids)
    return result
