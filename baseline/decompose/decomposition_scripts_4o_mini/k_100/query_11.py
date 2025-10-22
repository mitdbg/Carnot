# Strategy for Query 11: American action films about security and surveillance also bullying (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve American action films about security and surveillance
    security_films = retrieve("American action films about security and surveillance", 200)
    
    # Step 2: Retrieve American action films about bullying
    bullying_films = retrieve("American action films about bullying", 200)
    
    # Step 3: Combine results using set operations
    # We are interested in films that are about both themes
    combined_results = security_films & bullying_films
    
    # Convert the results to a list for output
    return list(combined_results)
