# Strategy for Query 12: Neogene mammals of Africa that are Odd-toed ungulates (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve documents related to "Neogene mammals of Africa"
    neogene_mammals_africa = retrieve("Neogene mammals of Africa", 200)
    
    # Step 2: Retrieve documents related to "Odd-toed ungulates"
    odd_toed_ungulates = retrieve("Odd-toed ungulates", 200)
    
    # Step 3: Find the intersection of the two sets to get Neogene mammals of Africa that are Odd-toed ungulates
    result = neogene_mammals_africa & odd_toed_ungulates
    
    # Convert the result to a list and return it
    return list(result)
