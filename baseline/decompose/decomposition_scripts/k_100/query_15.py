# Strategy for Query 15: Snakes of Asia that are Fauna of Oceania and Reptiles of the Philippines (No Truncation)
def execute_query(retrieve):
    # Retrieve broad candidate sets with generous k for intersections (5x target_k = 500)
    k_intersect = 500
    
    # 1) Candidates for "Snakes of Asia"
    snakes_asia = retrieve("Snakes of Asia", k_intersect)
    
    # 2) Candidates for "Fauna of Oceania"
    fauna_oceania = retrieve("Fauna of Oceania", k_intersect)
    
    # 3) Candidates for "Reptiles of the Philippines"
    reptiles_ph = retrieve("Reptiles of the Philippines", k_intersect)
    
    # Final result: items that are in all three categories
    result_set = snakes_asia & fauna_oceania & reptiles_ph
    
    # Return as a list (do not truncate)
    return list(result_set)
