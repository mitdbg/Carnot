# Strategy for Query 5: Holarctic and North American desert fauna and also Vertebrates of Belize (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents for both parts of the query
    holarctic_fauna_docs = retrieve("Holarctic desert fauna", 200)  # Retrieve a larger set for intersection
    north_american_fauna_docs = retrieve("North American desert fauna", 200)  # Retrieve a larger set for intersection
    belize_vertebrates_docs = retrieve("Vertebrates of Belize", 200)  # Retrieve a larger set for union
    
    # Step 2: Perform set operations
    # Intersection to find documents related to both Holarctic and North American desert fauna
    desert_fauna_intersection = holarctic_fauna_docs & north_american_fauna_docs
    
    # Union with Belize vertebrates
    final_docs = desert_fauna_intersection | belize_vertebrates_docs
    
    # Step 3: Return the results as a list
    return list(final_docs)
