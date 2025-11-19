# Strategy for Query 8: find me, Cisuralian animals, Paleozoic insects of Asia, or Carboniferous animals of Asia. (No Truncation)
def execute_query(retrieve):
    # Retrieving different categories of documents based on the query
    cisuralian_animals = retrieve("Cisuralian animals", 200)  # Larger k for intersections
    paleozoic_insects_asia = retrieve("Paleozoic insects of Asia", 200)  # Larger k for intersections
    carboniferous_animals_asia = retrieve("Carboniferous animals of Asia", 150)  # 1.5x target_k for union
    
    # Performing set operations based on the query
    result = cisuralian_animals | paleozoic_insects_asia | carboniferous_animals_asia  # Union of all sets
    
    # Return the result as a list
    return list(result)
