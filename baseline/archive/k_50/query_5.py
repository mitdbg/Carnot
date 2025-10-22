# Strategy for Query 5: Holarctic and North American desert fauna and also Vertebrates of Belize (No Truncation)
def execute_query(retrieve):
    # Retrieve documents related to "Holarctic and North American desert fauna"
    query1 = "Holarctic and North American desert fauna"
    results1 = retrieve(query1, 100)  # Get more than needed to correlate later
    
    # Retrieve documents related to "Vertebrates of Belize"
    query2 = "Vertebrates of Belize"
    results2 = retrieve(query2, 100)  # Get more than needed to correlate later
    
    # Combining results: the intersection of results1 and results2
    result = results1.intersection(results2)
    
    return result
