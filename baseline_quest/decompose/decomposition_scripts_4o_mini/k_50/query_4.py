# Strategy for Query 4: Aquatic animals from South America that are found in Victoria(Australia) (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve aquatic animals that are from South America
    south_american_aquatic_animals = retrieve("aquatic animals from South America", 100)  # Use a larger k to get more results
    
    # Step 2: Retrieve aquatic animals that are found in Victoria, Australia
    victoria_aquatic_animals = retrieve("aquatic animals found in Victoria, Australia", 100)  # Use a larger k to get more results
    
    # Step 3: Find the intersection of both sets
    result_set = south_american_aquatic_animals & victoria_aquatic_animals
    
    # Return the final result as a list
    return list(result_set)
