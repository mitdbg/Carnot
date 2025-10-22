# Strategy for Query 2: what are Novels by Robert B. Parker that are not set in Massachusetts (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve all novels by Robert B. Parker
    novels_by_parker = retrieve("Novels by Robert B. Parker", 50)
    
    # Step 2: Retrieve all novels set in Massachusetts
    novels_in_massachusetts = retrieve("Novels by Robert B. Parker set in Massachusetts", 50)
    
    # Step 3: Calculate the difference to find novels not set in Massachusetts
    novels_not_in_massachusetts = novels_by_parker - novels_in_massachusetts
    
    # Step 4: Return the result as a list
    result_list = list(novels_not_in_massachusetts)
    return result_list
