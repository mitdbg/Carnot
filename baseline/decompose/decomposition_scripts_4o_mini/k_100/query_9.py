# Strategy for Query 9: romance films from New Zealand (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve all romance films
    romance_films = retrieve("romance films", 150)
    
    # Step 2: Retrieve all films from New Zealand
    new_zealand_films = retrieve("films from New Zealand", 150)
    
    # Step 3: Intersect the two sets to find romance films specifically from New Zealand
    romance_nz_films = romance_films & new_zealand_films
    
    # Returning the result as a list
    return list(romance_nz_films)
