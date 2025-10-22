# Strategy for Query 8: find me, Cisuralian animals, Paleozoic insects of Asia, or Carboniferous animals of Asia. (No Truncation)
def execute_query(retrieve):
    # Decompose the query into three main retrieval tasks and combine with unions.
    # Use a conservative k (2x target_k) for union queries to allow overlap and coverage.
    k = 200
    
    # 1) Cisuralian animals (include a "fauna" variant)
    cis1 = retrieve("Cisuralian animals", k)
    cis2 = retrieve("Cisuralian fauna", k)
    cisuralian = cis1 | cis2
    
    # 2) Paleozoic insects of Asia (include British spelling and short variants)
    paleo1 = retrieve("Paleozoic insects of Asia", k)
    paleo2 = retrieve("Palaeozoic insects of Asia", k)
    paleo3 = retrieve("Paleozoic insects Asia", k)
    paleozoic_insects_asia = paleo1 | paleo2 | paleo3
    
    # 3) Carboniferous animals of Asia (include "fauna" and short variants)
    carb1 = retrieve("Carboniferous animals of Asia", k)
    carb2 = retrieve("Carboniferous fauna of Asia", k)
    carb3 = retrieve("Carboniferous animals Asia", k)
    carboniferous_asia = carb1 | carb2 | carb3
    
    # Final result: union of the three sets (do not truncate)
    result_set = cisuralian | paleozoic_insects_asia | carboniferous_asia
    
    # Return as a list
    return list(result_set)
