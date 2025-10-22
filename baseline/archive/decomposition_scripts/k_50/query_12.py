# Strategy for Query 12: Neogene mammals of Africa that are Odd-toed ungulates (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve documents related to Neogene mammals of Africa
    neogene_mammals_africa = retrieve("Neogene mammals of Africa", k_final)
    
    # Step 2: Retrieve documents related to Odd-toed ungulates
    odd_toed_ungulates = retrieve("Odd-toed ungulates", k_final)
    
    # Step 3: Combine the results using intersection to find documents that meet both criteria
    final_doc_ids = neogene_mammals_africa.intersection(odd_toed_ungulates)
    
    # Convert the final set of document IDs to a list
    result = list(final_doc_ids)
    return result
