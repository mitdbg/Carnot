# Strategy for Query 20: Books about Brunei, Malaysia or historical books about the Qing dynasty (No Truncation)
def execute_query(retrieve):
    k_final = 50
    brunei_books = retrieve("Books about Brunei", k_final)
    malaysia_books = retrieve("Books about Malaysia", k_final)
    qing_dynasty_books = retrieve("historical books about the Qing dynasty", k_final)
    
    final_doc_ids = brunei_books.union(malaysia_books).union(qing_dynasty_books)
    result = list(final_doc_ids)
    return result
