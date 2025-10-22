# Strategy for Query 6: Arecaceae that are trees of Indo-China (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents related to Arecaceae
    query_1 = "Arecaceae"
    results_1 = retrieve(query_1, 100)  # Retrieve more than needed to account for filtering
    
    # Step 2: Filter those to only trees
    query_2 = "trees"
    results_2 = retrieve(query_2, 100)  # Again, retrieve more to ensure we cover overlaps
    
    # Step 3: Retrieve documents specifically for Indo-China
    query_3 = "Indo-China"
    results_3 = retrieve(query_3, 100)
    
    # Combine results using set intersections
    results = results_1 & results_2 & results_3
        
    return results
