# Strategy for Query 12: Neogene mammals of Africa that are Odd-toed ungulates (No Truncation)
def execute_query(retrieve):
    # Retrieve candidate documents for "Neogene mammals of Africa" and for "odd-toed ungulates (Perissodactyla) in Africa",
    # then intersect the two sets to get Neogene African mammals that are odd-toed ungulates.
    # Use substantially larger k for retrievals involved in intersections (here 5x target_k = 500).
    # Use moderately large k for family-level / synonym queries that will be unioned (200).
    
    # Large retrievals for intersection candidates
    neogene_core = retrieve("Neogene mammals Africa", 500)
    neogene_miocene = retrieve("Miocene mammals Africa", 500)
    neogene_pliocene = retrieve("Pliocene mammals Africa", 500)
    
    # Combine Neogene-related queries (union)
    neogene_candidates = neogene_core | neogene_miocene | neogene_pliocene
    
    # Perissodactyl / odd-toed ungulate queries
    perissodactyla_core = retrieve("Perissodactyla Africa", 500)           # main taxon-level retrieval (large for intersection)
    odd_toed_generic = retrieve("odd-toed ungulates Africa", 500)
    # Family-level queries with moderately large k to broaden coverage (will be unioned)
    equidae = retrieve("Equidae Africa", 200)
    rhinocerotidae = retrieve("Rhinocerotidae Africa", 200)
    tapiridae = retrieve("Tapiridae Africa", 200)
    chalicotheriidae = retrieve("Chalicotheriidae Africa", 200)
    # Also include queries that couple Perissodactyla + Neogene to capture documents explicitly mentioning both
    perissodactyla_neogene = retrieve("Perissodactyla Neogene Africa", 500)
    odd_toed_neogene = retrieve("odd-toed ungulates Neogene Africa", 500)
    
    # Union of perissodactyl-related results
    perissodactyl_candidates = (
        perissodactyla_core
        | odd_toed_generic
        | equidae
        | rhinocerotidae
        | tapiridae
        | chalicotheriidae
        | perissodactyla_neogene
        | odd_toed_neogene
    )
    
    # Final intersection: Neogene African mammals AND odd-toed ungulates
    final_set = neogene_candidates & perissodactyl_candidates
    
    # Return as a list (do not truncate)
    return list(final_set)
