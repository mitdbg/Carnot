#!/usr/bin/env python3
# execute_all.py for k=50 - RETRIEVAL ONLY (NO TRUNCATION)

import json
import sys
import os
from tqdm import tqdm

# Add project root to path to allow importing retriever
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from retrieve import initialize_retriever, retrieve

# Import all generated strategy functions
from query_1 import execute_query as execute_query_1
from query_2 import execute_query as execute_query_2
from query_3 import execute_query as execute_query_3
from query_4 import execute_query as execute_query_4
from query_5 import execute_query as execute_query_5
from query_6 import execute_query as execute_query_6
from query_7 import execute_query as execute_query_7
from query_8 import execute_query as execute_query_8
from query_9 import execute_query as execute_query_9
from query_10 import execute_query as execute_query_10
from query_11 import execute_query as execute_query_11
from query_12 import execute_query as execute_query_12
from query_13 import execute_query as execute_query_13
from query_14 import execute_query as execute_query_14
from query_15 import execute_query as execute_query_15
from query_16 import execute_query as execute_query_16
from query_17 import execute_query as execute_query_17
from query_18 import execute_query as execute_query_18
from query_19 import execute_query as execute_query_19
from query_20 import execute_query as execute_query_20

# --- Main Execution ---
def main():
    # Initialize tools once
    initialize_retriever()

    # Get the list of queries from the gold file
    gold_file_path = "../../../train_subset.jsonl"
    with open(gold_file_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    output_path = "pred_retrieved_set_ops.jsonl"
    if os.path.exists(output_path):
        os.remove(output_path) # Clear old results

    # Map query index to its strategy function
    strategy_functions = [
        execute_query_1,
        execute_query_2,
        execute_query_3,
        execute_query_4,
        execute_query_5,
        execute_query_6,
        execute_query_7,
        execute_query_8,
        execute_query_9,
        execute_query_10,
        execute_query_11,
        execute_query_12,
        execute_query_13,
        execute_query_14,
        execute_query_15,
        execute_query_16,
        execute_query_17,
        execute_query_18,
        execute_query_19,
        execute_query_20,
    ]
    
    num_to_process = len(strategy_functions)

    for i in tqdm(range(num_to_process), desc="Executing all query strategies for k=50"):
        query_data = queries[i]
        query_text = query_data["query"]
        strategy_func = strategy_functions[i]
        
        final_docs = strategy_func(retrieve)
        
        prediction = {"query": query_text, "docs": final_docs}
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")
            
    print(f"\n--- All strategies executed for k=50. ---")
    print(f"Final predictions (no truncation) are ready in {output_path}") 

if __name__ == "__main__":
    main()
