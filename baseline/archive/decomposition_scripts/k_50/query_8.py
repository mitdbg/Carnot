# Strategy for Query 8: find me, Cisuralian animals, Paleozoic insects of Asia, or Carboniferous animals of Asia. (No Truncation)
def execute_query(retrieve):
    k_final = 50
    
    # Step 1: Retrieve Cisuralian animals
    cisuralian_animals = retrieve("Cisuralian animals", k_final)
    
    # Step 2: Retrieve Paleozoic insects of Asia
    paleozoic_insects_asia = retrieve("Paleozoic insects of Asia", k_final)
    
    # Step 3: Retrieve Carboniferous animals of Asia
    carboniferous_animals_asia = retrieve("Carboniferous animals of Asia", k_final)
    
    # Step 4: Combine results using set operations (union)
    final_doc_ids = cisuralian_animals | paleozoic_insects_asia | carboniferous_animals_asia
    
    # The result must be the full list of all found document IDs.
    result = list(final_doc_ids)
    return result
