# Strategy for Query 3: Endemic flora of Australia, Malaysia,and Fiji (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve the endemic flora of Australia
    australia_flora = retrieve("endemic flora of Australia", k_final)
    
    # Retrieve the endemic flora of Malaysia
    malaysia_flora = retrieve("endemic flora of Malaysia", k_final)
    
    # Retrieve the endemic flora of Fiji
    fiji_flora = retrieve("endemic flora of Fiji", k_final)
    
    # Combine the results using set operations
    final_doc_ids = australia_flora.union(malaysia_flora).union(fiji_flora)
    
    # Output the final list of document IDs
    result = list(final_doc_ids)
    return result
