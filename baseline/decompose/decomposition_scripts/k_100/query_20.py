# Strategy for Query 20: Books about Brunei, Malaysia or historical books about the Qing dynasty (No Truncation)
def execute_query(retrieve):
    # Retrieve books about Brunei, Malaysia, and historical books about the Qing dynasty,
    # then combine them with a union (|). Use a higher k (2x target_k = 200) for coverage.
    brunei = retrieve("books about Brunei", 200)
    malaysia = retrieve("books about Malaysia", 200)
    qing = retrieve("historical books about the Qing dynasty", 200)
    
    result = brunei | malaysia | qing
    # Return as a list (do not truncate)
    return list(result)
