# Strategy for Query 3: Endemic flora of Australia, Malaysia,and Fiji (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents related to endemic flora of each region
    australia_docs = retrieve("endemic flora of Australia", 20)  # Adjust k based on expected overlap
    malaysia_docs = retrieve("endemic flora of Malaysia", 20)
    fiji_docs = retrieve("endemic flora of Fiji", 20)
        
    # Step 2: Combine the results using set operations
    combined_results = australia_docs & malaysia_docs & fiji_docs
    return combined_results
