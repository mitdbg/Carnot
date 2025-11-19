# Strategy for Query 5: Holarctic and North American desert fauna and also Vertebrates of Belize (No Truncation)
def execute_query(retrieve):
    # Retrieve documents matching each part of the query
    holarctic_docs = retrieve("Holarctic desert fauna", 100)
    north_american_docs = retrieve("North American desert fauna", 100)
    vertebrates_belize_docs = retrieve("Vertebrates of Belize", 100)
    
    # Combine results using set operations
    combined_docs = (holarctic_docs | north_american_docs) & vertebrates_belize_docs
    
    # Return the final set of document IDs
    return list(combined_docs)
