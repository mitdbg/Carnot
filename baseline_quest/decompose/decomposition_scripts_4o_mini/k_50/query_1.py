# Strategy for Query 1: Stoloniferous plants or crops originating from Bolivia (No Truncation)
def execute_query(retrieve):
    # Retrieve documents about stoloniferous plants
    plants_query = "stoloniferous plants"
    plants_ids = retrieve(plants_query, 100)
    
    # Retrieve documents about crops originating from Bolivia
    crops_query = "crops from Bolivia"
    crops_ids = retrieve(crops_query, 100)
    
    # Combine results using union set operation
    results_ids = plants_ids | crops_ids
    
    # Convert the results to a list to return
    return list(results_ids)
