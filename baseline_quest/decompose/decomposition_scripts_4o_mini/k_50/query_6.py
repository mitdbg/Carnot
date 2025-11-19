# Strategy for Query 6: Arecaceae that are trees of Indo-China (No Truncation)
def execute_query(retrieve):
    # Define subqueries based on the main query
    subquery1 = "Arecaceae"
    subquery2 = "trees"
    subquery3 = "Indo-China"
    
    # Retrieve documents for each subquery
    results1 = retrieve(subquery1, 60)  # Retrieve more to ensure we can filter down later
    results2 = retrieve(subquery2, 60)
    results3 = retrieve(subquery3, 60)
    
    # Perform set operations to combine the results
    # Only keep documents that are found in both results1 and results2
    intermediate_results = results1 & results2
    
    # Now filter those based on results3
    final_results = intermediate_results & results3
    
    # Convert final results to a list and return 
    return list(final_results)
