# Strategy for Query 6: Arecaceae that are trees of Indo-China (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve all documents related to Arecaceae
    query1 = "Arecaceae"
    results1 = retrieve(query1, 150)  # using a larger k for better chance of getting relevant documents
    
    # Step 2: Retrieve all documents related to trees
    query2 = "trees"
    results2 = retrieve(query2, 150)  # using a larger k for better chance of getting relevant documents
    
    # Step 3: Retrieve all documents related to Indo-China
    query3 = "Indo-China"
    results3 = retrieve(query3, 150)  # using a larger k for better chance of getting relevant documents
    
    # Step 4: Intersect results1 and results2 to find Arecaceae that are trees
    results_trees = results1 & results2
    
    # Step 5: Intersect the previous result with results3 to find Arecaceae that are trees of Indo-China
    final_results = results_trees & results3
    
    # Step 6: Convert the set to a list to return
    return list(final_results)
