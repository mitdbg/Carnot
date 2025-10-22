# Strategy for Query 14: find me Epistemology books (No Truncation)
def execute_query(retrieve):
    # Step 1: Retrieve books related to "Epistemology"
    epistemology_books = retrieve("Epistemology books", 150)
    
    # Step 2: Retrieve general philosophy books
    general_philosophy_books = retrieve("Philosophy books", 150)
    
    # Step 3: Find the intersection of Epistemology books and Philosophy books
    epistemology_in_philosophy = epistemology_books & general_philosophy_books
    
    # Step 4: Retrieve books related to "Epistemology" not specific to Philosophy
    non_philosophical_epistemology_books = epistemology_books - epistemology_in_philosophy
    
    # Step 5: Combine both lists
    return list(epistemology_in_philosophy | non_philosophical_epistemology_books)
