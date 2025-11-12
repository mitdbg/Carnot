#!/usr/bin/env python3

import json
import sys
import os
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decompose.k_scripts.retrieve_limited_verbose import initialize_retriever, retrieve

query_num = 10
from decompose.k_scripts.query_10.query_10 import execute_query

# --- Main Execution ---
def main():
    # 1. Initialize tools once (your actual retriever)
    initialize_retriever()

    gold_file_path = "../../data/train_subset.jsonl"
    try:
        with open(gold_file_path, "r", encoding="utf-8") as f:
            queries = [json.loads(line) for line in f if line.strip()]
            
        QUERY_TEXT = queries[query_num-1]["query"]
        print(f"Loaded Query {query_num} text: '{QUERY_TEXT}'")
        
    except FileNotFoundError:
        print(f"Error: Gold file not found at '{gold_file_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading query from gold file: {e}")
        return

    # 3. Define the iteration parameters
    k_values = list(range(100, 1001, 100)) # 100, 200, ..., 1000 (10 values)
    k_combinations = list(product(k_values, k_values)) # 100 total combinations
    
    OUTPUT_PATH = f"pred_query_{query_num}_varying_k.jsonl"
    
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH) # Clear old results

    print(f"--- Starting execution of Query {query_num} for {len(k_combinations)} combinations ---")
    
    # 4. Loop through all (k1, k2) pairs and execute the strategy
    for k1, k2 in tqdm(k_combinations, desc=f"Executing Query {query_num} Strategy"):
        
        docs_1, docs_2, final_docs = execute_query(retrieve, k1, k2)
        
        # Create the prediction dictionary
        prediction = {
            "query": QUERY_TEXT, 
            "k1": k1,
            "docs_1": docs_1,
            "k2": k2, 
            "docs_2": docs_2,
            "docs": final_docs
        }
        
        # Write the result to the JSONL file
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")
            
    print("\n------------------------------------------------------------")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
