# Strategy for Query 17: 1997 anime films or slayer films (No Truncation)
def execute_query(retrieve):
    # Retrieve documents for "1997 anime films" (use a generous k for unions: ~2x target_k)
    k_union = 200
    queries_1997_anime = [
        "1997 anime films",
        "anime films 1997",
        "anime films released in 1997",
        "1997 Japanese animated films",
    ]
    
    set_1997_anime = set()
    for q in queries_1997_anime:
        set_1997_anime |= retrieve(q, k_union)
    
    # Retrieve documents for "slayer films" (use similar k for coverage)
    queries_slayer = [
        "slayer films",
        "Slayer (film)",
        "films titled Slayer",
        "slayer movie",
        "movies about slayers",
        "films with 'Slayer' in the title",
    ]
    
    set_slayer = set()
    for q in queries_slayer:
        set_slayer |= retrieve(q, k_union)
    
    # Final result: union of 1997 anime films OR slayer films
    final_set = set_1997_anime | set_slayer
    
    # Return as a list (do not truncate)
    return list(final_set)
