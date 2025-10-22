# Strategy for Query 11: American action films about security and surveillance also bullying (No Truncation)
def execute_query(retrieve):
    # Retrieve broad candidate sets. Use a large k (≈5x target_k = 500) for intersections to maximize recall.
    american_action = retrieve("American action films", 500)
    surveillance = retrieve("films about surveillance OR surveillance films OR surveillance society", 500)
    security = retrieve("films about security OR private security OR security guard OR corporate security", 500)
    bullying = retrieve("films about bullying OR bullying behavior OR harassment OR school bullying OR workplace bullying", 500)
    
    # Combine surveillance and security (union) to cover either theme
    sec_or_surv = surveillance | security
    
    # Final intersection: American action films that are about (security OR surveillance) AND also about bullying
    final_set = american_action & sec_or_surv & bullying
    
    # Return as a list (do not truncate)
    return list(final_set)
