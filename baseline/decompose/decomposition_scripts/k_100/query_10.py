# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve):
    # retrieve broad lists (use large k for intersections to avoid missing relevant docs)
    year_1912 = retrieve("1912 films", 500)
    set_in_england = retrieve("films set in England", 500)
    
    # direct/phrased queries (smaller k ok for unions)
    direct_phrase = retrieve("1912 films set in England", 200)
    british_1912 = retrieve("1912 British films", 500)
    uk_set = retrieve("films set in the United Kingdom", 200)
    
    # Combine:
    # - Core: intersection of films from 1912 and films set in England
    # - Add direct matches and overlaps with British 1912 or UK-set pages
    results = (year_1912 & set_in_england) | direct_phrase | (year_1912 & british_1912) | (year_1912 & uk_set & set_in_england)
    
    # Return as a list (do not truncate)
    return list(results)
