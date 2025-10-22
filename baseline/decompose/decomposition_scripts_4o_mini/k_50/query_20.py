# Strategy for Query 20: Books about Brunei, Malaysia or historical books about the Qing dynasty (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents about Brunei
    brunei_docs = retrieve("Books about Brunei", 50)
    
    # Step 2: Retrieve documents about Malaysia
    malaysia_docs = retrieve("Books about Malaysia", 50)
    
    # Step 3: Retrieve historical books about the Qing dynasty
    qing_docs = retrieve("Historical books about the Qing dynasty", 50)
    
    # Step 4: Combine results using set operations
    # Union of Brunei and Malaysia
    brunei_malaysia_union = brunei_docs | malaysia_docs
    
    # Combine with Qing dynasty historical books
    final_results = brunei_malaysia_union | qing_docs
    
    # Return the final result as a list
    return list(final_results)
