# Strategy for Query 4: Aquatic animals from South America that are found in Victoria(Australia) (No Truncation)
def execute_query(retrieve):
    # Retrieve candidate sets for "aquatic animals from South America" using several focused queries (unions)
    sa_q1 = retrieve("aquatic species native to South America", k=200)
    sa_q2 = retrieve("freshwater fish of South America", k=200)
    sa_q3 = retrieve("South American aquatic mammals and cetaceans", k=200)
    sa_q4 = retrieve("South American amphibians and aquatic frogs", k=200)
    sa_candidates = set(sa_q1) | set(sa_q2) | set(sa_q3) | set(sa_q4)
    
    # Retrieve candidate sets for "found in Victoria (Australia)" using several focused queries (unions)
    vic_q1 = retrieve("aquatic species found in Victoria Australia", k=200)
    vic_q2 = retrieve("freshwater species in Victoria Australia", k=200)
    vic_q3 = retrieve("marine species and coastal fauna Victoria Australia", k=200)
    vic_q4 = retrieve("introduced aquatic species in Victoria Australia", k=200)
    vic_candidates = set(vic_q1) | set(vic_q2) | set(vic_q3) | set(vic_q4)
    
    # To ensure high recall for intersection operations, retrieve broader high-k sets and intersect them
    sa_large = set(retrieve("aquatic animals from South America", k=500))
    vic_large = set(retrieve("aquatic animals found in Victoria Australia", k=500))
    high_recall_intersection = sa_large & vic_large
    
    # Combine the union-of-subqueries intersection with the high-recall intersection
    approx_intersection = sa_candidates & vic_candidates
    final_set = approx_intersection | high_recall_intersection
    
    # Return results as a list (do not truncate to target_k)
    return list(final_set)
