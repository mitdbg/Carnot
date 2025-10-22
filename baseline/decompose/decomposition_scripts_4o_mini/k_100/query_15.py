# Strategy for Query 15: Snakes of Asia that are Fauna of Oceania and Reptiles of the Philippines (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve snakes of Asia
    asia_snakes = retrieve("Snakes of Asia", 200)
    
    # Step 2: Retrieve fauna of Oceania
    oceania_fauna = retrieve("Fauna of Oceania", 200)
    
    # Step 3: Retrieve reptiles of the Philippines
    philippines_reptiles = retrieve("Reptiles of the Philippines", 200)
    
    # Step 4: Find snakes that are both fauna of Oceania and reptiles of the Philippines
    fauna_and_reptiles = (asia_snakes & oceania_fauna) | (asia_snakes & philippines_reptiles)
    
    # Combine the results
    return list(fauna_and_reptiles)
