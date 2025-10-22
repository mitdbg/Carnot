# Strategy for Query 6: Arecaceae that are trees of Indo-China (No Truncation)
def execute_query(retrieve):
    # Retrieve broad set of palms / Arecaceae (use large k because we'll intersect)
    a_set = retrieve("Arecaceae OR palm OR palms OR \"family Arecaceae\"", 500)
    
    # Retrieve documents describing tree habit / arborescent palms (large k for intersection)
    tree_set = retrieve("tree OR \"palm tree\" OR arborescent OR \"tree-like\" OR \"single-stem\" OR trunk OR \"arborescent palm\"", 500)
    
    # Retrieve region documents for Indo-China and constituent countries (unions can use a smaller multiple)
    region_indochina = retrieve("Indochina OR \"Indo-China\" OR Indochinese", 200)
    region_vietnam = retrieve("Vietnam", 200)
    region_laos = retrieve("Laos", 200)
    region_cambodia = retrieve("Cambodia", 200)
    region_thailand = retrieve("Thailand", 200)
    region_myanmar = retrieve("Myanmar OR Burma", 200)
    region_malay_pen = retrieve("Peninsular Malaysia OR \"Malay Peninsula\"", 200)
    
    # Union of all region-related hits
    region_set = (region_indochina | region_vietnam | region_laos |
                  region_cambodia | region_thailand | region_myanmar |
                  region_malay_pen)
    
    # A focused high-precision retrieval combining family, habit and region (also large k)
    focused = retrieve(
        "Arecaceae AND (tree OR arborescent OR \"palm tree\" OR \"arborescent palm\" OR \"tree-like\") "
        "AND (Indochina OR \"Indo-China\" OR Vietnam OR Laos OR Cambodia OR Thailand OR Myanmar OR \"Peninsular Malaysia\")",
        500
    )
    
    # Final set: intersection of palms, tree-habit, and Indo-China region, plus any in the focused retrieval
    final_set = (a_set & tree_set & region_set) | focused
    
    # Return as a list (do not truncate to target_k)
    return list(final_set)
