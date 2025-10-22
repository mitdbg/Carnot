# Strategy for Query 18: what are Australian travel books (No Truncation)
def execute_query(retrieve):
    # Retrieve documents related to Australian travel books
    travel_books_query = "Australian travel books"
    general_travel_query = "travel books"
    
    # Retrieve sets of documents for each query
    australian_travel_books = retrieve(travel_books_query, 200)  # Larger k for a better chance to get all needed documents
    general_travel_books = retrieve(general_travel_query, 100)  # A smaller k for general travel books
    
    # Combine results
    final_results = australian_travel_books | general_travel_books  # Union of both sets
    
    # Convert the final results to a list
    return list(final_results)
    final_results_list
    return result
