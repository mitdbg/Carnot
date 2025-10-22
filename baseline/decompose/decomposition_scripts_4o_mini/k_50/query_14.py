# Strategy for Query 14: find me Epistemology books (No Truncation)
def execute_query(retrieve):
    # First, retrieve all documents related to Epistemology books
    epistemology_books = retrieve("Epistemology books", 50)
    
    # Since there's no specific set operation mentioned, no additional queries are necessary
    # Return the results as a list
    return list(epistemology_books)
