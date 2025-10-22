# Strategy for Query 13: Films set in Pittsburgh that are LGBT-related romance (No Truncation)
def execute_query(retrieve):
    # Retrieve films set in Pittsburgh (large k because we'll intersect)
    pittsburgh = retrieve("films set in Pittsburgh OR movies set in Pittsburgh OR Pittsburgh-set films", 500)
    
    # Retrieve LGBT-related films using several related queries and union them (moderate k per query)
    lgbt_gay = retrieve("LGBT-related films OR gay films OR queer films OR LGBT films", 200)
    lesbian = retrieve("lesbian films OR lesbian romance films", 200)
    trans = retrieve("transgender films OR trans films", 200)
    bisexual = retrieve("bisexual films", 200)
    lgbt_union = lgbt_gay | lesbian | trans | bisexual
    
    # Retrieve romance films (large k because we'll intersect)
    romance = retrieve("romance films OR romantic films OR romantic comedy OR romantic drama OR love story film", 500)
    
    # Primary intersection: films that are set in Pittsburgh AND LGBT-related AND romance
    intersection_result = pittsburgh & lgbt_union & romance
    
    # Also retrieve direct/explicit matches for "LGBT romance films set in Pittsburgh" to boost recall (large k)
    direct_specific = retrieve("LGBT romance films set in Pittsburgh OR gay romance Pittsburgh OR lesbian romance Pittsburgh OR queer romance Pittsburgh", 500)
    
    # Final result: union of intersection and direct specific retrievals (do not truncate)
    final_results = intersection_result | direct_specific
    
    # Return as list
    return list(final_results)
