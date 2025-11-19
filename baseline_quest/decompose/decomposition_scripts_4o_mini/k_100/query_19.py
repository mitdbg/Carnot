# Strategy for Query 19: Set in the Edo period but not Japanese historical films (No Truncation)
def execute_query(retrieve):
    # Retrieve documents pertaining to the Edo period
    edo_period_docs = retrieve("Edo period", 200)
    
    # Retrieve documents pertaining to Japanese historical films
    japanese_historical_films_docs = retrieve("Japanese historical films", 100)
    
    # Calculate the final result: Edo period documents that are not Japanese historical films
    result_docs = edo_period_docs - japanese_historical_films_docs
    
    # Convert the set to a list and return
    return list(result_docs)
