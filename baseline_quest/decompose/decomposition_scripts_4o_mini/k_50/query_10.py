# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve):
    # First, we retrieve the documents related to "1912 films"
    films_1912 = retrieve("1912 films", 100)
    
    # Then, we retrieve the documents related to "films set in England"
    films_in_england = retrieve("films set in England", 100)
    
    # Now, we combine the results using set intersection, since we want films that are both from 1912 and set in England
    final_results = films_1912 & films_in_england
    
    # Finally, we return the results as a list
    return list(final_results)
