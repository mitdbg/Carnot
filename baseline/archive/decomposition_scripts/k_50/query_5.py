# Strategy for Query 5: Holarctic and North American desert fauna and also Vertebrates of Belize (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve documents for each part of the query
    holarctic_fauna = retrieve("Holarctic desert fauna", k_final)
    north_american_fauna = retrieve("North American desert fauna", k_final)
    vertebrates_belize = retrieve("Vertebrates of Belize", k_final)
    
    # Step 2: Combine results using set operations
    final_doc_ids = holarctic_fauna.union(north_american_fauna).union(vertebrates_belize)
    
    # Full list of all found document IDs
    result = list(final_doc_ids)
    return result
