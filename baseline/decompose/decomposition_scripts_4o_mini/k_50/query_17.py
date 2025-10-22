# Strategy for Query 17: 1997 anime films or slayer films (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents related to "1997 anime films"
    anime_1997 = retrieve("1997 anime films", 100)
    
    # Step 2: Retrieve documents related to "slayer films"
    slayer_films = retrieve("slayer films", 100)
    
    # Step 3: Combine results using set operations
    # We want the union of both sets
    results = anime_1997 | slayer_films
    
    # Return the results as a list
    return list(results)
