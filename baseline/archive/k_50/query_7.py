# Strategy for Query 7: Pinnipeds of Antarctica or South America or Marine fauna of Antarctica (No Truncation)
def execute_query(retrieve):
    # Decompose the query into smaller subqueries
    subquery1 = "Pinnipeds of Antarctica"
    subquery2 = "Pinnipeds of South America"
    subquery3 = "Marine fauna of Antarctica"
        
    # Retrieve top-k results for each subquery
    results1 = retrieve(subquery1, 25)  # Adjust k for OR operation
    results2 = retrieve(subquery2, 25)  # Adjust k for OR operation
    results3 = retrieve(subquery3, 25)  # Adjust k for OR operation
        
    # Combine results using set union
    result = results1 | results2 | results3  # Union of all results
    
    return result
