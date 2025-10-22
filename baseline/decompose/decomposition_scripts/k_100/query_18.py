# Strategy for Query 18: what are Australian travel books (No Truncation)
def execute_query(retrieve):
    # Decompose "what are Australian travel books" into focused retrieval subtasks,
    # then combine results using set operations.
    # target_k = 100 (given). Use larger k for intersections/differences (5x => 500),
    # and moderately larger k for unions (1.5x-2x => 150-200).
    
    # Broad union candidates (use ~2x target_k)
    s_travel_aust     = set(retrieve("travel books Australia", 200))
    s_aust_travel     = set(retrieve("Australian travel books", 200))
    s_guide_books     = set(retrieve("Australia travel guide book", 200))
    s_travelogues     = set(retrieve("Australia travelogue travel book", 200))
    s_road_trip_books = set(retrieve("Australian road trip book travel", 200))
    s_travel_writing  = set(retrieve("Australian travel writing memoir book", 200))
    s_photo_books     = set(retrieve("Australian travel photography book", 150))
    s_lonely_planet   = set(retrieve("Lonely Planet Australia guidebook", 150))
    s_nonfiction      = set(retrieve("nonfiction travel books Australia", 200))
    
    # High-confidence intersection: items that strongly match "book" + "Australia" + "travel"
    # Use much larger k (5x target_k) to ensure candidates are present for intersection.
    s_keywords_intersection = set(retrieve("book Australia travel", 500))
    s_intersection_ref1     = set(retrieve("Australia travel guide book", 500))
    s_intersection_ref2     = set(retrieve("Australian travel writing", 500))
    
    high_confidence = s_keywords_intersection & s_intersection_ref1 & s_intersection_ref2
    
    # Exclude likely-irrelevant categories (e.g., pure fiction novels not presented as travel books)
    # Use large k for the difference step as well.
    s_fiction = set(retrieve("Australia novel fiction not travel", 500))
    
    # Combine everything:
    union_all = (
        s_travel_aust
        | s_aust_travel
        | s_guide_books
        | s_travelogues
        | s_road_trip_books
        | s_travel_writing
        | s_photo_books
        | s_lonely_planet
        | s_nonfiction
    )
    
    # Final set: include union and high-confidence intersection, exclude fiction-like results
    final_set = (union_all | high_confidence) - s_fiction
    
    # Return as a list (do not truncate to target_k)
    return list(final_set)
