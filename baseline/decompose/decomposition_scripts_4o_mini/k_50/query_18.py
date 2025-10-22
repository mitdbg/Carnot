# Strategy for Query 18: what are Australian travel books (No Truncation)
def execute_query(retrieve):
    # Define the initial query
    query_travel = "Australian travel"
    query_books = "books"
    
    # Retrieve documents related to Australian travel
    travel_docs = retrieve(query_travel, 100)  # Use a higher k to ensure we have enough results
    # Retrieve documents related to books
    books_docs = retrieve(query_books, 100)
    
    # Perform set operation to find books related to Australian travel
    result_docs = travel_docs & books_docs
    
    # Return the final result as a list
    return list(result_docs)
