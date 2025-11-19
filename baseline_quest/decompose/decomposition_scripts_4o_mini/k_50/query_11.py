# Strategy for Query 11: American action films about security and surveillance also bullying (No Truncation)
def execute_query(retrieve):
    # Breaking down the query into components
    security_and_surveillance_query = "American action films about security and surveillance"
    bullying_query = "American action films about bullying"
    
    # Retrieving documents for both queries
    security_and_surveillance_docs = retrieve(security_and_surveillance_query, 100)  # Adjust k based on expected set size
    bullying_docs = retrieve(bullying_query, 100)  # Adjust k based on expected set size
    
    # Performing set operations
    final_results = security_and_surveillance_docs & bullying_docs  # Intersection
    
    # Converting to a list for final output
    return list(final_results)
