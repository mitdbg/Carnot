# Strategy for Query 7: Pinnipeds of Antarctica or South America or Marine fauna of Antarctica (No Truncation)
def execute_query(retrieve):
    # Retrieve documents for each part of the OR query.
    # Use a generous k for each retrieval (2x target_k = 200) to allow for overlap and ensure coverage.
    pinnipeds_antarctica = set(retrieve("Pinnipeds of Antarctica", 200))
    pinnipeds_south_america = set(retrieve("Pinnipeds of South America", 200))
    marine_fauna_antarctica = set(retrieve("Marine fauna of Antarctica", 200))
    
    # Combine with union (OR)
    results = pinnipeds_antarctica | pinnipeds_south_america | marine_fauna_antarctica
    
    # Return as a list (do not truncate)
    return list(results)
