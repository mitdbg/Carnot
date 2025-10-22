# Strategy for Query 20: Books about Brunei, Malaysia or historical books about the Qing dynasty (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):
        # Step 1: Retrieve books about Brunei
        brunei_books = retrieve("Books about Brunei", 50)
    
        # Step 2: Retrieve books about Malaysia
        malaysia_books = retrieve("Books about Malaysia", 50)
    
        # Step 3: Retrieve historical books about the Qing dynasty
        qing_books = retrieve("Historical books about the Qing dynasty", 50)
    
        # Step 4: Combine results using set operations
        result = brunei_books | malaysia_books | qing_books  # Union of all sets
    
        return result
    return result
