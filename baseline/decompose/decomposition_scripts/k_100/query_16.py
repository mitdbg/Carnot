# Strategy for Query 16: Brazilian fantasy films,or shot in Amazonas (No Truncation)
def execute_query(retrieve):
    # Strategy: form two sets and take their union:
    # 1) Brazilian fantasy films (various phrasings)
    # 2) Films shot in Amazonas (various phrasings)
    # Use k = 200 (2x target_k=100) for each retrieve call to allow overlap and good coverage.
    
    k_union = 200
    
    # Queries to capture Brazilian fantasy films (English and Portuguese variants)
    queries_brazil_fantasy = [
        "Brazilian fantasy films",
        "Brazilian fantasy film",
        "fantasy films Brazil",
        "fantasia brasileira filme",
        "filmes de fantasia brasileiros",
        "Brazilian fantasy cinema",
    ]
    
    # Queries to capture films shot in Amazonas (English and Portuguese variants)
    queries_amazonas = [
        "films shot in Amazonas",
        "shot in Amazonas",
        "filmed in Amazonas",
        "films shot in Amazonas, Brazil",
        "filmed in Amazonas state",
        "films shot in the Amazon rainforest",
        "filmes filmados no Amazonas",
        "filmado no Amazonas",
    ]
    
    # Retrieve and union results for Brazilian fantasy films
    brazil_results = [retrieve(q, k_union) for q in queries_brazil_fantasy]
    brazil_set = set().union(*brazil_results) if brazil_results else set()
    
    # Retrieve and union results for films shot in Amazonas
    amazonas_results = [retrieve(q, k_union) for q in queries_amazonas]
    amazonas_set = set().union(*amazonas_results) if amazonas_results else set()
    
    # Final result: union of the two sets (Brazilian fantasy films OR shot in Amazonas)
    final_set = brazil_set | amazonas_set
    
    # Return as a list (do not truncate)
    return list(final_set)
