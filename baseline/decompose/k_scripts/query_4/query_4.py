# Strategy for Query 4: Aquatic animals from South America that are found in Victoria(Australia) (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve aquatic animals from South America
    south_america_animals = retrieve("aquatic animals from South America", 200)
    
    # Step 2: Retrieve animals found in Victoria, Australia
    victoria_animals = retrieve("aquatic animals found in Victoria Australia", 200)
    
    # Step 3: Find the intersection of both sets to get aquatic animals that are from South America and found in Victoria
    result_set = south_america_animals & victoria_animals
    
    # Convert the result set to a list
    return list(result_set)