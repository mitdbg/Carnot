# Strategy for Query 1: Stoloniferous plants or crops originating from Bolivia (No Truncation)
def execute_query(retrieve):
    # First, retrieve documents for "stoloniferous plants"
    results_stoloniferous = retrieve("stoloniferous plants", 30)  # Retrieve more as this is a broader category
        
    # Next, retrieve documents for "crops originating from Bolivia"
    results_crops_bolivia = retrieve("crops originating from Bolivia", 30)  # Retrieve more to ensure coverage
        
    # Combine results using set union for 'or' operation
    combined_results = results_stoloniferous | results_crops_bolivia
        
    return combined_results
