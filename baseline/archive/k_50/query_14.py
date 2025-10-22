# Strategy for Query 14: find me Epistemology books (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):
        # First retrieve the top documents related to Epistemology
        epistemology_docs = retrieve("Epistemology books", 60)  # Retrieve more than needed to ensure we get close to target_k
        # For the final result, we want to limit it to the target_k
        result = epistemology_docs
        return result
    return result
