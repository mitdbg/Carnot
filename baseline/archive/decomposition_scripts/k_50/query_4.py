# Strategy for Query 4: Aquatic animals from South America that are found in Victoria(Australia) (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve aquatic animals from South America
    south_american_aquatic_animals = retrieve("aquatic animals from South America", k_final)
    
    # Step 2: Retrieve animals found in Victoria, Australia
    victoria_aquatic_animals = retrieve("aquatic animals found in Victoria, Australia", k_final)
    
    # Step 3: Combine results using intersection to find common aquatic animals
    final_doc_ids = south_american_aquatic_animals.intersection(victoria_aquatic_animals)
    
    # Step 4: Convert to list for return
    result = list(final_doc_ids)
    return result
