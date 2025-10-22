# Strategy for Query 2: what are Novels by Robert B. Parker that are not set in Massachusetts (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve all novels by Robert B. Parker
    all_novels = retrieve("Novels by Robert B. Parker", 200)
    
    # Step 2: Retrieve novels set in Massachusetts
    massachusetts_novels = retrieve("Novels by Robert B. Parker set in Massachusetts", 200)
    
    # Step 3: Find novels by Robert B. Parker that are not set in Massachusetts
    novels_not_in_massachusetts = all_novels - massachusetts_novels
    
    # Convert the result to a list
    return list(novels_not_in_massachusetts)
