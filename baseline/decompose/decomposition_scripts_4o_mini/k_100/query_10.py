# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve):
    # First, retrieve a broad set of documents related to 1912 films
    films_1912 = retrieve("1912 films", 200)
    
    # Now, retrieve a set of documents related to films set in England
    films_set_in_england = retrieve("films set in England", 200)
    
    # Find the intersection of both sets to get films that are in both categories
    films_1912_set_in_england = films_1912 & films_set_in_england
    
    # Convert the result to a list for output
    return list(films_1912_set_in_england)
