# Strategy for Query 8: find me, Cisuralian animals, Paleozoic insects of Asia, or Carboniferous animals of Asia. (No Truncation)
def execute_query(retrieve):
    # Retrieve Cisuralian animals
    cisuralian_animals = retrieve("Cisuralian animals", 50)
        
    # Retrieve Paleozoic insects of Asia
    paleozoic_insects_asia = retrieve("Paleozoic insects of Asia", 50)
        
    # Retrieve Carboniferous animals of Asia
    carboniferous_animals_asia = retrieve("Carboniferous animals of Asia", 50)
        
    # Combine results using set union
    result = cisuralian_animals | paleozoic_insects_asia | carboniferous_animals_asia
        
    return result
