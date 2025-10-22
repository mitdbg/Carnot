# Strategy for Query 8: find me, Cisuralian animals, Paleozoic insects of Asia, or Carboniferous animals of Asia. (No Truncation)
def execute_query(retrieve):
    # Retrieve Cisuralian animals
    cisuralian_animals = retrieve("Cisuralian animals", 60)
    
    # Retrieve Paleozoic insects of Asia
    paleozoic_insects_asia = retrieve("Paleozoic insects of Asia", 60)
    
    # Retrieve Carboniferous animals of Asia
    carboniferous_animals_asia = retrieve("Carboniferous animals of Asia", 60)
    
    # Combine the results using set operations: Union of all three sets
    final_results = cisuralian_animals | paleozoic_insects_asia | carboniferous_animals_asia
    
    # Return the final results as a list
    return list(final_results)
