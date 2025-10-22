#!/usr/bin/env python3

"""
Uses DSPy to generate modular Python strategy files for each QUEST query.
Generates a master `execute_all.py` script in each output directory
to run the full pipeline using a separate retriever module and set operations,
**without** reranking.
"""
import os
import json
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

# --- Configuration ---
# NOTE: The path uses ../ to access the file from a directory above (assuming the script is run from a subfolder).
INPUT_QUERIES_PATH = "../train_subset.jsonl"
OUTPUT_STRATEGIES_DIR = "decomposition_scripts"

# Get API key from environment variable
# If the key is not found, an error will occur during DSPy configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

LLM_MODEL = 'gpt-4o-mini'
TARGET_KS = [20] # 50, 100

# --- Data Loading ---
@dataclass
class QuestQuery:
    query: str
    docs: List[str]
    original_query: str
    metadata: Optional[Dict[str, Any]] = None

def load_quest_queries(path: str) -> List[QuestQuery]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(QuestQuery(**{k: v for k, v in data.items() if k in QuestQuery.__annotations__}))
    return queries

# --- DSPy Module for Query Decomposition (WITHOUT RERANKING) ---
class QueryDecomposerWithPython(dspy.Signature):
    query = dspy.InputField(desc="A natural language retrieval task with implicit set operations.")
    target_k = dspy.InputField(desc="The target number of final documents to return.")
    
    output = dspy.OutputField(
        desc="A Python program that decomposes the query and combines the results using set operations (&, |, -). "
             "The program must adhere to the following rules:\n"
             "1. Call `retrieve_chunks_for_query(query_string, k)` for retrieval, where `k` is the number of chunks.\n"
             "2. Perform set operations on the document IDs (the dictionary keys).\n"
             "3. The result of the set operation is a list of document IDs.\n"
             "4. **Crucially, truncate the final list of document IDs to `target_k` documents.** Example: `result = list(final_doc_ids)[:target_k]`.\n"
             "**IMPORTANT: Write ONLY the Python code *inside* the function.** Do NOT write `def execute_query(...)`. Your response should start with `k_final = ...` or similar logic.\n"
             "Wrap the code block in <python> and </python>."
    )

class QuestQueryDecomposer(dspy.Module):
    # Updated example to remove reranking
    few_shot_examples = [
        dspy.Example(
            query="Trees of South Africa that are also in the south-central Pacific",
            target_k=50,
            output="""<python>
k_final = 50
k_retrieve_intersection = k_final * 10
results_a = retrieve_chunks_for_query('Trees of South Africa', k=k_retrieve_intersection)
results_b = retrieve_chunks_for_query('trees in the south-central Pacific', k=k_retrieve_intersection)
doc_ids_a = set(results_a.keys())
doc_ids_b = set(results_b.keys())
final_doc_ids = doc_ids_a & doc_ids_b # Perform set operation
# Truncate the list of IDs
result = list(final_doc_ids)[:k_final] 
</python>"""
        ),
    ]

    def __init__(self):
        super().__init__()
        # FIX APPLIED: Set temperature to 1.0 to resolve the gpt-5 model compatibility error.
        self.generate_code = dspy.Predict(QueryDecomposerWithPython, n=1, temperature=1.0)

    def forward(self, query: str, target_k: int):
        # NOTE: Demos is used instead of examples for backwards compatibility with some dspy versions
        return self.generate_code(query=query, target_k=target_k, Demos=self.few_shot_examples)

# --- Template for the master execution script (NO RERANKING) ---
EXECUTE_ALL_TEMPLATE = """#!/usr/bin/env python3
# execute_all.py for k={k} - RETRIEVAL ONLY

import json
import sys
import os
from tqdm import tqdm

# Add project root to path to allow importing retriever
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Note: reranker import is removed
from retriever import initialize_retriever, retrieve_chunks_for_query

# Import all generated strategy functions
{import_statements}

# --- Main Execution ---
def main():
    # Initialize tools once
    initialize_retriever()
    # Note: initialize_reranker() is removed

    # Get the list of queries from the gold file
    gold_file_path = "../../../train_subset.jsonl"
    with open(gold_file_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    output_path = "pred_retrieved_set_ops.jsonl" # Renamed output file
    if os.path.exists(output_path):
        os.remove(output_path) # Clear old results

    # Map query index to its strategy function
    strategy_functions = [
        {strategy_list}
    ]
    
    num_to_process = len(strategy_functions)

    for i in tqdm(range(num_to_process), desc="Executing all query strategies for k={k}"):
        query_data = queries[i]
        query_text = query_data["query"]
        strategy_func = strategy_functions[i]
        
        # Pass only the retriever function, as rerank is no longer used
        final_docs = strategy_func(retrieve_chunks_for_query)
        
        prediction = {{"query": query_text, "docs": final_docs}}
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\\n")
            
    print(f"\\n--- All strategies executed for k={k}. ---")
    print(f"Final predictions (no reranking) are ready in {{output_path}}")

if __name__ == "__main__":
    main()
"""

def generate_strategies():
    if not OPENAI_API_KEY:
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        return

    # Set the API key using the environment variable value.
    try:
        lm = dspy.LM(
            f'openai/{LLM_MODEL}', 
            temperature=1.0, 
            max_tokens=16000, 
            api_key=OPENAI_API_KEY
        )
        dspy.configure(lm=lm)
        print(f"Successfully configured dspy.LM with model: {LLM_MODEL}")
    except Exception as e:
        print(f"Error configuring DSPy language model: {e}")
        return

    queries = load_quest_queries(INPUT_QUERIES_PATH)
    
    queries_to_process = queries

    decomposer = QuestQueryDecomposer()
    
    for k in TARGET_KS:
        output_dir_for_k = os.path.join(OUTPUT_STRATEGIES_DIR, f"k_{k}")
        os.makedirs(output_dir_for_k, exist_ok=True)
        
        import_statements = []
        strategy_list = []

        print(f"\n--- Generating Python strategies for k={k} into '{output_dir_for_k}/' (No Reranking) ---")
        for i, quest_query in enumerate(queries_to_process):
            print(f"  Processing query {i+1}/{len(queries_to_process)}...")
            try:
                result = decomposer(query=quest_query.query, target_k=k)
                # Ensure only the core Python code is extracted
                python_code = result.output.strip().replace("<python>", "").replace("</python>", "").strip()
            except Exception as e:
                print(f"    ERROR generating strategy for query {i+1}: {e}")
                print("    Skipping this query.")
                continue

            filename = f"query_{i+1}.py"
            filepath = os.path.join(output_dir_for_k, filename)
            with open(filepath, "w", encoding="utf-8") as f_out:
                f_out.write(f"# Strategy for Query {i+1}: {quest_query.query} (No Reranking)\n")
                # Removed 'rerank' argument from function signature
                f_out.write(f'def execute_query(retrieve_chunks_for_query):\n') 
                f_out.write("    " + python_code.replace("\n", "\n    "))
                f_out.write(f'\n    return result\n')
            
            import_statements.append(f"from query_{i+1} import execute_query as execute_query_{i+1}")
            strategy_list.append(f"execute_query_{i+1},")
        
        # Write the master execution script
        master_script_content = EXECUTE_ALL_TEMPLATE.format(
            k=k,
            import_statements="\n".join(import_statements),
            strategy_list="\n        ".join(strategy_list)
        )
        with open(os.path.join(output_dir_for_k, "execute_all.py"), "w", encoding="utf-8") as f_master:
            f_master.write(master_script_content)

    print("\n--- All strategies and execution scripts generated successfully! ---")

if __name__ == "__main__":
    generate_strategies()