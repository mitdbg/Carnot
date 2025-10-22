# Strategy for Query 17: 1997 anime films or slayer films (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):
        # Retrieve documents for '1997 anime films'
        anime_films = retrieve("1997 anime films", 30)
    
        # Retrieve documents for 'slayer films'
        slayer_films = retrieve("slayer films", 30)
    
        # Combine results using set union
        result = anime_films.union(slayer_films)
    
        return result
    return result
