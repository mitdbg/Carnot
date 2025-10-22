# Strategy for Query 9: romance films from New Zealand (No Truncation)
def execute_query(retrieve):
    # First, retrieve films that are classified as romance
    romance_films = retrieve("romance films", 100)
    
    # Next, retrieve films that are classified as from New Zealand
    new_zealand_films = retrieve("films from New Zealand", 100)
    
    # Combine the results to get only the New Zealand romance films
    new_zealand_romance_films = romance_films & new_zealand_films
    
    # Return the results as a list
    return list(new_zealand_romance_films)