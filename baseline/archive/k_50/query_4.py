# Strategy for Query 4: Aquatic animals from South America that are found in Victoria(Australia) (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve aquatic animals from South America
    south_america_aquatic = retrieve("aquatic animals from South America", 100)  # Get more to ensure we have enough
    
    # Step 2: Retrieve animals found in Victoria, Australia
    victoria_aquatic = retrieve("aquatic animals found in Victoria, Australia", 100)
    
    # Step 3: Find the intersection of both sets to get animals from South America that are found in Victoria
    result = south_america_aquatic & victoria_aquatic
        
    return result
