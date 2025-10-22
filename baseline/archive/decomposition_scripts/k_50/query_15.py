# Strategy for Query 15: Snakes of Asia that are Fauna of Oceania and Reptiles of the Philippines (No Truncation)
def execute_query(retrieve):
    k_final = 50
    asia_snakes = retrieve("Snakes of Asia", k_final)
    oceania_fauna = retrieve("Fauna of Oceania", k_final)
    philippines_reptiles = retrieve("Reptiles of the Philippines", k_final)
    
    # Combining the results with set operations
    final_doc_ids = asia_snakes & oceania_fauna & philippines_reptiles
    
    result = list(final_doc_ids)
    return result
