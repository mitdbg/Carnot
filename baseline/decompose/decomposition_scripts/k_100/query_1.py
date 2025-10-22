# Strategy for Query 1: Stoloniferous plants or crops originating from Bolivia (No Truncation)
def execute_query(retrieve):
    # Retrieve stoloniferous plants (use a large k because we'll later intersect)
    stolon_queries = [
        "stoloniferous plants",
        "plants that spread by stolons",
        "plants with runners (stolons)",
        "stoloniferous species list",
        "plants that produce stolons",
        "stoloniferous crop"
    ]
    stolon_candidates = set()
    for q in stolon_queries:
        stolon_candidates |= set(retrieve(q, 500))  # 5x target_k to maximize recall for intersection
    
    # Retrieve plants/crops originating in Bolivia (also large k for intersection)
    bolivia_queries = [
        "plants native to Bolivia",
        "endemic to Bolivia plants",
        "plants originating in Bolivia",
        "Bolivian native plants",
        "crops originating in Bolivia",
        "plants from Bolivia"
    ]
    bolivia_candidates = set()
    for q in bolivia_queries:
        bolivia_candidates |= set(retrieve(q, 500))  # 5x target_k
    
    # Intersection: stoloniferous AND originating in Bolivia
    final_set = stolon_candidates & bolivia_candidates
    
    # Augment with direct queries that combine both concepts (use moderately large k)
    direct_queries = [
        "stoloniferous plants Bolivia",
        "plants that spread by stolons Bolivia",
        "Bolivia plants with runners",
        "Bolivian stoloniferous crop",
    ]
    for q in direct_queries:
        final_set |= set(retrieve(q, 200))  # ~2x target_k for union-style retrieval
    
    # Return results as a list (do not truncate)
    return list(final_set)
