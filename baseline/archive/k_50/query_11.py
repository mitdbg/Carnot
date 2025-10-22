# Strategy for Query 11: American action films about security and surveillance also bullying (No Truncation)
def execute_query(retrieve):
    def execute_query(retrieve):
        # Step 1: Retrieve documents for action films about security and surveillance
        query1 = "American action films about security and surveillance"
        action_security_docs = retrieve(query1, 60)  # retrieve more to ensure we get enough results
        
        # Step 2: Retrieve documents for action films about bullying
        query2 = "American action films about bullying"
        action_bullying_docs = retrieve(query2, 60)  # retrieve more to ensure we get enough results
    
        # Step 3: Intersect both sets to find common documents
        result = action_security_docs.intersection(action_bullying_docs)
        
        return result
    return result
