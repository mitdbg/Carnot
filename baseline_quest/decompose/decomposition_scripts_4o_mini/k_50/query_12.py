# Strategy for Query 12: Neogene mammals of Africa that are Odd-toed ungulates (No Truncation)
def execute_query(retrieve):
    # Decomposing the query into subqueries and performing set operations
    
    # Step 1: Retrieve Neogene mammals of Africa
    neogene_mammals_africa = retrieve("Neogene mammals of Africa", 100)
    
    # Step 2: Retrieve Odd-toed ungulates
    odd_toed_ungulates = retrieve("Odd-toed ungulates", 100)
    
    # Step 3: Perform an intersection to find common documents
    result = neogene_mammals_africa & odd_toed_ungulates
    
    # Convert the result to a list and return
    return list(result)
