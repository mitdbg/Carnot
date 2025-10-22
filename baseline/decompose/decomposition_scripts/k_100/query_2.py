# Strategy for Query 2: what are Novels by Robert B. Parker that are not set in Massachusetts (No Truncation)
def execute_query(retrieve):
    # Retrieve a broad set of candidate documents that are novels by Robert B. Parker,
    # using a large k (5x target_k = 500) since we'll perform intersections/differences.
    parker_primary = set(retrieve("novels by Robert B. Parker", 500))
    # Add a secondary union to improve recall (union can use smaller multiple like 2x target_k = 200)
    parker_secondary = set(retrieve("Robert B. Parker books", 200))
    parker_all = parker_primary | parker_secondary
    
    # Retrieve novels set in Massachusetts (broad) using large k for reliable intersection/difference
    ma_primary = set(retrieve("novels set in Massachusetts", 500))
    # Secondary variant to improve recall
    ma_secondary = set(retrieve("books set in Massachusetts", 200))
    ma_all = ma_primary | ma_secondary
    
    # Identify which Parker novels are set in Massachusetts (intersection) and remove them
    parker_set_in_ma = parker_all & ma_all
    parker_not_in_ma = parker_all - parker_set_in_ma
    
    # Return final results as a list (do not truncate)
    return list(parker_not_in_ma)
