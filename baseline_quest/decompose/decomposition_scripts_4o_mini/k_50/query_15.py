# Strategy for Query 15: Snakes of Asia that are Fauna of Oceania and Reptiles of the Philippines (No Truncation)
def execute_query(retrieve):
    # Decomposing the query into sub-queries and using set operations to combine results
    asia_snakes = retrieve("Snakes of Asia", 100)
    oceania_fauna = retrieve("Fauna of Oceania", 100)
    philippines_reptiles = retrieve("Reptiles of the Philippines", 100)
    
    # Performing set operations
    result = asia_snakes & oceania_fauna - philippines_reptiles
    
    # Returning the result as a list
    return list(result)
