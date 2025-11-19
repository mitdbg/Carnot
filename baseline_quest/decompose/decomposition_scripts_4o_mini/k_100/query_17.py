# Strategy for Query 17: 1997 anime films or slayer films (No Truncation)
def execute_query(retrieve):
    # Retrieve documents related to 1997 anime films
    anime_1997 = retrieve("1997 anime films", 200)
    
    # Retrieve documents related to slayer films
    slayer_films = retrieve("slayer films", 200)
    
    # Combine results using set operations
    return list(anime_1997 | slayer_films)  # Union of both sets