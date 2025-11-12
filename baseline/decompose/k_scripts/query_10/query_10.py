# # Strategy for Query 10: 1912 films set in England (No Truncation)
# def execute_query(retrieve, k1, k2):
#     """
#     Executes the set-intersection strategy for Query 10 and returns 
#     the final document list along with the sizes of the two initial retrieval sets.
    
#     Returns:
#         tuple: (list[str], int, int) -> (final_docs, size_films_1912, size_films_set_in_england)
#     """
#     # First, retrieve a broad set of documents related to 1912 films
#     films_1912 = retrieve("1912 films", k1)
    
#     # Now, retrieve a set of documents related to films set in England
#     # films_set_in_england = retrieve("films set in England", k2)
#     films_set_in_england = retrieve("English films", k2)
    
#     # Find the intersection of both sets to get films that are in both categories
#     films_1912_set_in_england = films_1912 & films_set_in_england
    
#     # Convert the result to a list for output and return sizes
#     return list(films_1912), list(films_set_in_england), list(films_1912_set_in_england)

from typing import Dict

# --- Strategy Definition ---

# Strategy for Query 10: 1912 films set in England (No Truncation)
def execute_query(retrieve, k1, k2):
    """
    Executes the set-intersection strategy for Query 10 and returns 
    the final documents (as a dict) along with the sizes of the two initial retrieval sets.
    
    Args:
        retrieve (function): The retrieve function from retriever.py
        k1 (int): Number of documents to retrieve for the first query
        k2 (int): Number of documents to retrieve for the second query

    Returns:
        tuple: (Dict[str, str], int, int) -> 
               (final_docs_dict, size_films_1912, size_films_set_in_england)
    """
    # First, retrieve a broad set of documents related to 1912 films
    # retrieve() returns a Dict[title, chunk_text]
    films_1912_dict = retrieve("1912 films", k1)
    
    # Now, retrieve a set of documents related to films set in England
    films_set_in_england_dict = retrieve("English films", k2)
    
    # Get the sets of titles (keys) from the dictionaries
    films_1912_titles = set(films_1912_dict.keys())
    films_set_in_england_titles = set(films_set_in_england_dict.keys())

    # Find the intersection of both title sets
    intersecting_titles = films_1912_titles & films_set_in_england_titles
    
    # Build the final dictionary with titles and their corresponding chunks
    # We can get the chunk text from either dictionary; films_1912_dict is fine.
    final_docs_dict = {
        title: films_1912_dict[title] 
        for title in intersecting_titles
    }
    
    # Return the final dictionary and the sizes of the initial sets
    return final_docs_dict, films_1912_dict, films_set_in_england_dict