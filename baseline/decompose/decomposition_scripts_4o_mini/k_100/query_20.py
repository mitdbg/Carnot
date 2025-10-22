# Strategy for Query 20: Books about Brunei, Malaysia or historical books about the Qing dynasty (No Truncation)
def execute_query(retrieve):
    # Retrieve documents about Brunei
    docs_brunei = retrieve("Books about Brunei", 200)
    
    # Retrieve documents about Malaysia
    docs_malaysia = retrieve("Books about Malaysia", 200)
    
    # Retrieve historical books about the Qing dynasty
    docs_qing_dynasty = retrieve("historical books about the Qing dynasty", 200)
    
    # Combine results: books about Brunei or Malaysia
    union_brunei_malaysia = docs_brunei | docs_malaysia
    
    # Combine with historical books about the Qing dynasty
    final_result = union_brunei_malaysia | docs_qing_dynasty
    
    # Return the result as a list
    return list(final_result)
