# Strategy for Query 9: romance films from New Zealand (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve all romance films
    romance_films = retrieve("romance films", 100)  # retrieve more than needed to ensure we have enough
        
    # Step 2: Retrieve films from New Zealand
    nz_films = retrieve("films from New Zealand", 100)  # same here to filter out romance later
    
    # Step 3: Intersect both sets to get romance films from New Zealand
    result = romance_films.intersection(nz_films)
        
    return result
