# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve top documents related to '1912 films'
    query1 = "1912 films"
    films_1912 = retrieve(query1, 100)  # Assume a high number for initial retrieval
    
    # Step 2: Retrieve top documents related to 'films set in England'
    query2 = "films set in England"
    films_set_in_england = retrieve(query2, 100)  # Assume a high number for initial retrieval
    
    # Step 3: Perform intersection to find films that are both from 1912 and set in England
    result = films_1912.intersection(films_set_in_england)
    
    return result
