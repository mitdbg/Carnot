# Strategy for Query 7: Pinnipeds of Antarctica or South America or Marine fauna of Antarctica (No Truncation)
def execute_query(retrieve):
    # Retrieve documents based on the query components
    antarctica_pinnipeds = retrieve("Pinnipeds of Antarctica", 50)
    south_america_pinnipeds = retrieve("Pinnipeds of South America", 50)
    antarctica_marine_fauna = retrieve("Marine fauna of Antarctica", 50)
    
    # Combine results using set operations
    result = antarctica_pinnipeds | south_america_pinnipeds | antarctica_marine_fauna
    
    # Return the results as a list
    return list(result)
