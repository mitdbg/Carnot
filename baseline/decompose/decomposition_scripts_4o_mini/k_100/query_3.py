# Strategy for Query 3: Endemic flora of Australia, Malaysia,and Fiji (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents related to endemic flora of Australia
    australia_docs = retrieve("endemic flora of Australia", 200)  # Higher k for better chances
    
    # Step 2: Retrieve documents related to endemic flora of Malaysia
    malaysia_docs = retrieve("endemic flora of Malaysia", 200)  # Higher k for better chances
    
    # Step 3: Retrieve documents related to endemic flora of Fiji
    fiji_docs = retrieve("endemic flora of Fiji", 200)  # Higher k for better chances
    
    # Step 4: Combine results using set operations
    result = australia_docs | malaysia_docs | fiji_docs  # Union of all documents
    
    # Step 5: Convert result to list and return
    return list(result)
