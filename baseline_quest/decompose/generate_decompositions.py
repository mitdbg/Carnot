#!/usr/bin/env python3

"""
Uses DSPy to generate modular Python strategy files for each QUEST query.
Generates a master `execute_all.py` script in each output directory
to run the full pipeline using a separate retriever module and set operations,
**without** reranking, and **without truncating the final result list**.

IMPROVEMENT: Refined the DSPy Signature and few-shot examples to strongly 
encourage significantly larger 'k' values (e.g., k=200 or more) for intersection 
and difference operations, thereby maximizing initial recall before set filtering.
"""
import os
import json
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

# --- Configuration ---
# NOTE: The path uses ../ to access the file from a directory above (assuming the script is run from a subfolder).
INPUT_QUERIES_PATH = "../data/train_subset3.jsonl"
OUTPUT_STRATEGIES_DIR = "decomposition_scripts_limited"

# Get API key from environment variable
# If the key is not found, an error will occur during DSPy configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = 'gpt-5-mini' # Can be changed to 'gpt-4o' for better results if budget permits
TARGET_KS = [100] # 20, 100

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

# --- DSPy Module for Query Decomposition (RETRIEVAL ONLY) ---
class QueryDecomposerWithPython(dspy.Signature):
    query = dspy.InputField(desc="A natural language retrieval task with implicit set operations.")
    target_k = dspy.InputField(desc="The target number of final documents to return. The final list will NOT be truncated to this k.")
    
    output = dspy.OutputField(
        desc="A Python program that decomposes the query into a sequence of smaller retrieval tasks and combines the results using set operations (&, |, -). The program must call the"
             "function retrieve(query_string, k) where k is the top-k documents to retrieve and returns a set of document IDs (str). "
             "**CRITICAL:** For set operations, especially **intersections (&)** and **differences (-) where a subset of documents is expected**, **k MUST be significantly larger** than the target_k (e.g., 4x or 5x the target_k, but keep it a round number like 100 or 200) to ensure a high chance of retrieving the documents needed for the final set. For **unions (|)**, k can be closer to the target_k, but still allow for overlap (e.g., 1.5x to 2x target_k)."
             "Return the results from set operations as a list. DO NOT truncate by the target_k."
             "Write ONLY the Python code *inside* the function.** Do NOT write `def execute_query(...)`.\n"
             "Wrap the code block in <python> and </python>."
    )

class QuestQueryDecomposer(dspy.Module):
    # Adjusted k values: 
    # Union (OR): k=100 for target_k=50 (1.5x to 2x)
    # Intersection (AND): k=200 for target_k=50 (4x)
    # Difference (BUT NOT): k=200 for target_k=50 (4x)
    few_shot_examples = [
        dspy.Example(
            query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
            target_k=50,
            output="""<python>
birds_of_kolombangara_ids = retrieve('Birds of Kolombangara', k=100)
western_province_birds_ids = retrieve('Birds of the Western Province (Solomon Islands)', k=100)
final_doc_ids = birds_of_kolombangara_ids | western_province_birds_ids
result = list(final_doc_ids)
return result
</python>"""
        ),
        dspy.Example(
            query="Trees of South Africa that are also in the south-central Pacific",
            target_k=50,
            output="""<python>
trees_sa_ids = retrieve('Trees of South Africa', k=200)
trees_scp_ids = retrieve('south-central Pacific', k=200)
final_doc_ids = trees_sa_ids & trees_scp_ids
result = list(final_doc_ids)
return result
</python>"""
        ),
        dspy.Example(
            query="2010's adventure films set in the Southwestern United States but not in California",
            target_k=50,
            output="""<python>
adventure_films_ids = retrieve('2010s adventure films', k=200)
sw_us_films_ids = retrieve('films set in the Southwestern United States', k=200)
california_films_ids = retrieve('films set in California', k=200)
inclusive_films = adventure_films_ids & sw_us_films_ids
final_doc_ids = inclusive_films - california_films_ids
result = list(final_doc_ids)
return result
</python>"""
        )
    ]

    def __init__(self):
        super().__init__()
        # Kept at 1.0, but consider lowering to 0.7 or 0.5 if the model ignores the k-sizing guidance.
        self.generate_code = dspy.Predict(QueryDecomposerWithPython, n=1, temperature=1.0) 

    def forward(self, query: str, target_k: int):
        # NOTE: Using Demos=... to pass the few-shot examples explicitly
        return self.generate_code(query=query, target_k=target_k, Demos=self.few_shot_examples)

# --- Template for the master execution script (NO RERANKING) ---
EXECUTE_ALL_TEMPLATE = """#!/usr/bin/env python3
# execute_all.py for k={k} - RETRIEVAL ONLY (NO TRUNCATION)

import json
import sys
import os
from tqdm import tqdm

# Add project root to path to allow importing retriever
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from retrieve import initialize_retriever, retrieve

# Import all generated strategy functions
{import_statements}

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
        {strategy_list}
    ]
    
    num_to_process = len(strategy_functions)

    for i in tqdm(range(num_to_process), desc="Executing all query strategies for k={k}"):
        query_data = queries[i]
        query_text = query_data["query"]
        strategy_func = strategy_functions[i]
        
        # The strategy function takes the retrieve function and returns a list of doc IDs.
        final_docs = strategy_func(retrieve)
        
        prediction = {{"query": query_text, "docs": final_docs}}
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\\n")
            
    print(f"\\n--- All strategies executed for k={k}. ---")
    print(f"Final predictions (no truncation) are ready in {{output_path}}") 

if __name__ == "__main__":
    main()
"""

def generate_strategies():
    if not OPENAI_API_KEY:
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        return

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

        print(f"\n--- Generating Python strategies for k={k} into '{output_dir_for_k}/' (No Truncation) ---")
        for i, quest_query in enumerate(queries_to_process):
            print(f"  Processing query {i+1}/{len(queries_to_process)}: {quest_query.query}")
            try:
                # The .output is a Prediction object; we need the text.
                result = decomposer(query=quest_query.query, target_k=k)
                # Ensure only the core Python code is extracted
                # Added strip() after replace for robustness
                python_code = result.output.strip().replace("<python>", "").replace("</python>", "").strip()
            except Exception as e:
                print(f"    ERROR generating strategy for query {i+1}: {e}")
                print("    Skipping this query.")
                continue

            filename = f"query_{i+1}.py"
            filepath = os.path.join(output_dir_for_k, filename)
            with open(filepath, "w", encoding="utf-8") as f_out:
                f_out.write(f"# Strategy for Query {i+1}: {quest_query.query} (No Truncation)\n")
                f_out.write(f'def execute_query(retrieve):\n') 
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