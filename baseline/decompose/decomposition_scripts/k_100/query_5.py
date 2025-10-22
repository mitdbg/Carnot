# Strategy for Query 5: Holarctic and North American desert fauna and also Vertebrates of Belize (No Truncation)
def execute_query(retrieve):
    # Retrieve candidate documents for "Holarctic fauna" (use large k because we'll intersect)
    h1 = retrieve("Holarctic fauna", 500)
    h2 = retrieve("Holarctic species", 500)
    h3 = retrieve("fauna of the Holarctic region", 500)
    # include Nearctic/Palearctic synonyms to broaden Holarctic coverage
    h4 = retrieve("Nearctic fauna", 500)
    h5 = retrieve("Palearctic fauna", 500)
    
    # Combine Holarctic-related results (union)
    holarctic_set = h1 | h2 | h3 | h4 | h5
    
    # Retrieve candidate documents for "North American desert fauna" (use large k for intersection)
    d1 = retrieve("North American desert fauna", 500)
    # Include major North American deserts to improve recall
    d2 = retrieve("Sonoran Desert fauna", 500)
    d3 = retrieve("Chihuahuan Desert fauna", 500)
    d4 = retrieve("Mojave Desert fauna", 500)
    d5 = retrieve("Great Basin Desert fauna", 500)
    
    # Combine North American desert results (union)
    north_american_desert_set = d1 | d2 | d3 | d4 | d5
    
    # Intersection: fauna that are both Holarctic-related and part of North American desert fauna
    holarctic_and_na_desert = holarctic_set & north_american_desert_set
    
    # Retrieve candidate documents for "Vertebrates of Belize"
    # For unions, smaller k (1.5x-2x target_k) is acceptable
    b1 = retrieve("Vertebrates of Belize", 200)
    b2 = retrieve("Belize vertebrates", 200)
    b3 = retrieve("fauna of Belize vertebrates", 200)
    # include common vertebrate groups in Belize to catch documents that list specific groups
    b4 = retrieve("Belize reptiles amphibians birds mammals", 200)
    b5 = retrieve("mammals birds reptiles Belize", 200)
    
    vertebrates_belize_set = b1 | b2 | b3 | b4 | b5
    
    # Final result: (Holarctic AND North American desert fauna) OR Vertebrates of Belize
    final_set = holarctic_and_na_desert | vertebrates_belize_set
    
    # Return as a list (do not truncate)
    return list(final_set)
