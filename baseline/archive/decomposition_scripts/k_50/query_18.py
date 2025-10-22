# Strategy for Query 18: what are Australian travel books (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve documents related to "Australian travel"
    australian_travel_docs = retrieve("Australian travel", k_final)
    
    # Step 2: Retrieve documents related to "travel books"
    travel_books_docs = retrieve("travel books", k_final)
    
    # Step 3: Combine the results using intersection to find common documents
    final_doc_ids = australian_travel_docs.intersection(travel_books_docs)
    
    # Step 4: Convert the final set of document IDs to a list
    result = list(final_doc_ids)
    return result
