# Strategy for Query 2: what are Novels by Robert B. Parker that are not set in Massachusetts (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve all novels by Robert B. Parker
    novels_by_parker = retrieve("Novels by Robert B. Parker", k_final)
    
    # Step 2: Retrieve all novels by Robert B. Parker set in Massachusetts
    massachusetts_novels = retrieve("Novels by Robert B. Parker set in Massachusetts", k_final)
    
    # Step 3: Compute the difference to find novels not set in Massachusetts
    final_doc_ids = novels_by_parker - massachusetts_novels
    
    # Step 4: Assign the result as a list
    result = list(final_doc_ids)
    return result
