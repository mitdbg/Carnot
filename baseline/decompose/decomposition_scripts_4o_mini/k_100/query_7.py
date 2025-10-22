# Strategy for Query 7: Pinnipeds of Antarctica or South America or Marine fauna of Antarctica (No Truncation)
def execute_query(retrieve):
    # Retrieve documents related to each part of the query
    ant_pinnipeds = retrieve("Pinnipeds of Antarctica", 150)  # k is significantly larger
    south_pinnipeds = retrieve("Pinnipeds of South America", 150)
    marine_fauna = retrieve("Marine fauna of Antarctica", 150)
    
    # Combine results using set operations
    result = ant_pinnipeds | south_pinnipeds | marine_fauna  # Union of all three sets
    return list(result)  # Convert the result to a list

