# Strategy for Query 2: what are Novels by Robert B. Parker that are not set in Massachusetts (No Truncation)
def execute_query(retrieve):
    # Retrieve novels by Robert B. Parker
    novels_by_parker = retrieve("Novels by Robert B. Parker", 100)  # Fetch more to filter down later
        
    # Retrieve novels set in Massachusetts
    massachusetts_novels = retrieve("Novels by Robert B. Parker set in Massachusetts", 100)
        
    # Set operations: Find novels not set in Massachusetts
    result = novels_by_parker - massachusetts_novels
        
    return result
