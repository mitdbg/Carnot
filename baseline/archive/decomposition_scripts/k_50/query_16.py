# Strategy for Query 16: Brazilian fantasy films,or shot in Amazonas (No Truncation)
def execute_query(retrieve):
    k_final = 50
    results_a = retrieve("Brazilian fantasy films", k_final)
    results_b = retrieve("films shot in Amazonas", k_final)
    final_doc_ids = results_a | results_b  # Union of both sets
    result = list(final_doc_ids)
    return result
