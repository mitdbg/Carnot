# Strategy for Query 12: Neogene mammals of Africa that are Odd-toed ungulates (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):  
        # Step 1: Retrieve documents for "Neogene mammals of Africa"
        neogene_mammals_africa = retrieve("Neogene mammals of Africa", 100)
    
        # Step 2: Retrieve documents for "Odd-toed ungulates"
        odd_toed_ungulates = retrieve("Odd-toed ungulates", 100)
    
        # Step 3: Perform intersection to find documents that are both Neogene mammals of Africa and Odd-toed ungulates
        result = neogene_mammals_africa.intersection(odd_toed_ungulates)
        
        return result
    return result
