# Strategy for Query 16: Brazilian fantasy films,or shot in Amazonas (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):  
        # Step 1: Retrieve documents related to Brazilian fantasy films
        brazil_fantasy_docs = retrieve("Brazilian fantasy films", k=50)
        
        # Step 2: Retrieve documents related to films shot in Amazonas
        amazon_docs = retrieve("films shot in Amazonas", k=50)
        
        # Step 3: Combine results using set operations
        # We are looking for documents that are either Brazilian fantasy films or shot in Amazonas
        result = brazil_fantasy_docs.union(amazon_docs)
        
        return result
    return result
