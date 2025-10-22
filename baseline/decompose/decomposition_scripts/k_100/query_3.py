# Strategy for Query 3: Endemic flora of Australia, Malaysia,and Fiji (No Truncation)
def execute_query(retrieve):
    # Retrieve multiple queries for each country and union their results.
    # Use a relatively large k for unions (2x target_k = 200) to capture overlap and variations.
    k_union = 200
    
    # Query variants to capture different phrasings and lists of endemic flora/plants
    australia_queries = [
        "endemic flora of Australia",
        "endemic plants Australia",
        "list of endemic Australian plants",
        "Australian endemic flora species list",
        "endemic vascular plants Australia",
    ]
    
    malaysia_queries = [
        "endemic flora of Malaysia",
        "endemic plants Malaysia",
        "list of endemic Malaysian plants",
        "Malaysian endemic flora species list",
        "endemic vascular plants Malaysia",
    ]
    
    fiji_queries = [
        "endemic flora of Fiji",
        "endemic plants Fiji",
        "list of endemic Fijian plants",
        "Fiji endemic flora species list",
        "endemic vascular plants Fiji",
    ]
    
    # Retrieve and union per country
    australia_sets = [retrieve(q, k_union) for q in australia_queries]
    australia_endemic = set().union(*australia_sets)
    
    malaysia_sets = [retrieve(q, k_union) for q in malaysia_queries]
    malaysia_endemic = set().union(*malaysia_sets)
    
    fiji_sets = [retrieve(q, k_union) for q in fiji_queries]
    fiji_endemic = set().union(*fiji_sets)
    
    # Final result: union of endemic flora for Australia, Malaysia, and Fiji
    final_set = australia_endemic | malaysia_endemic | fiji_endemic
    
    # Return as a list (do not truncate)
    return list(final_set)
