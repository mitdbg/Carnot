# Strategy for Query 9: romance films from New Zealand (No Truncation)
def execute_query(retrieve):
    # Retrieve candidate sets for "romance" genre (use ~2x target_k per union component)
    r_romance_1 = retrieve("romance film", 200)
    r_romance_2 = retrieve("romantic film", 200)
    r_romance_3 = retrieve("romance movie", 200)
    r_romance_4 = retrieve("romantic movie", 200)
    r_romance_5 = retrieve("love story film", 200)
    
    romance_union = r_romance_1 | r_romance_2 | r_romance_3 | r_romance_4 | r_romance_5
    
    # Retrieve candidate sets for "New Zealand" origin (use ~2x target_k per union component)
    r_nz_1 = retrieve("New Zealand film", 200)
    r_nz_2 = retrieve("films from New Zealand", 200)
    r_nz_3 = retrieve("New Zealand movie", 200)
    r_nz_4 = retrieve("NZ film", 200)
    
    nz_union = r_nz_1 | r_nz_2 | r_nz_3 | r_nz_4
    
    # Retrieve higher-recall, combined queries (use significantly larger k for intersection-style precision)
    r_precise_1 = retrieve("romance film New Zealand OR 'New Zealand romance film' OR 'romantic New Zealand film'", 500)
    r_precise_2 = retrieve("romantic comedy New Zealand OR 'New Zealand romantic comedy' OR 'NZ romantic comedy'", 500)
    r_precise_3 = retrieve("love story New Zealand OR 'New Zealand love story'", 500)
    
    # Combine: intersect the broad genre and country unions, and include high-recall precise queries
    candidates = (romance_union & nz_union) | r_precise_1 | r_precise_2 | r_precise_3
    
    # Return as a list (do not truncate)
    return list(candidates)
