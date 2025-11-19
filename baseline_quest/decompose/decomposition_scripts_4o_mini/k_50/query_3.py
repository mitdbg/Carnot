# Strategy for Query 3: Endemic flora of Australia, Malaysia,and Fiji (No Truncation)
def execute_query(retrieve):
    # Retrieve documents related to endemic flora from each country
    australia_docs = retrieve("endemic flora of Australia", 20)
    malaysia_docs = retrieve("endemic flora of Malaysia", 20)
    fiji_docs = retrieve("endemic flora of Fiji", 20)
    
    # Compute the intersection (common documents) of all three sets
    common_docs = australia_docs & malaysia_docs & fiji_docs
    
    # Compute the union (all unique documents) of the three sets
    all_docs = australia_docs | malaysia_docs | fiji_docs
    
    # Final result: documents that are either common in all or unique across any
    result_set = common_docs | (all_docs - common_docs)
    
    # Return result as a list
    result_list = list(result_set)
    
    return result_list
