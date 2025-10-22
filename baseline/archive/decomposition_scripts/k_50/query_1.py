# Strategy for Query 1: Stoloniferous plants or crops originating from Bolivia (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Retrieve documents related to "stoloniferous plants"
    stoloniferous_plants_docs = retrieve("stoloniferous plants", k_final)
    
    # Retrieve documents related to "crops originating from Bolivia"
    crops_bolivia_docs = retrieve("crops originating from Bolivia", k_final)
    
    # Combine the results using set union
    final_doc_ids = stoloniferous_plants_docs | crops_bolivia_docs
    
    # Convert the final set to a list
    result = list(final_doc_ids)
    return result
