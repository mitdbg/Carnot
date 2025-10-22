# Strategy for Query 7: Pinnipeds of Antarctica or South America or Marine fauna of Antarctica (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve documents for each sub-query
    pinnipeds_antarctica = retrieve("Pinnipeds of Antarctica", k_final)
    pinnipeds_south_america = retrieve("Pinnipeds of South America", k_final)
    marine_fauna_antarctica = retrieve("Marine fauna of Antarctica", k_final)
    
    # Combine results using set union
    final_doc_ids = pinnipeds_antarctica | pinnipeds_south_america | marine_fauna_antarctica
    
    # Convert the final set of document IDs to a list
    result = list(final_doc_ids)
    return result
