# Baseline Experiments with QUEST

## Experiment Setup
1. Indexed with Chroma (`indexing/index_chroma.py`)
    - 512 tokens with 80-token overlap / index only first 512 tokens (same as QUEST)
    - Embedded in batches of 256 chunks
    - Embedding model: `bge-small-en-v1.5`

2. Decompose (optional)
    - Use gpt-4o-mini to decompose the query into subqueries connected with operators
        - E.g. "Stoloniferous plants or crops originating from Bolivia" -> retrieve("crops from Bolivia", 100) | retrieve("stoloniferous plants", 100)
        - E.g. "Neogene mammals of Africa that are Odd-toed ungulates" -> retrieve("Neogene mammals of Africa", 100) & retrieve("Odd-toed ungulates", 100)
        - Examples decomposition python files: `decompose/k_scripts/query_9` and `decompose/k_scripts/query_10`
    - Generate decompositions with `decompose/generate_decompositions.py`

3. Retrieval (@ k)
    - Retrieve with query / subquery (vector similarity)
        - If indexed entired documented, retrieves the top 200 most likely chunks, then maps them to the document
        - If indexed first 512 tokens only, retrieves the top k most likely chunks
    - Retrieve after decomposition with query / subquery
        - `decompose/execute_decompositions.py` executes all of the generated decomposition pythons scripts from step 2 and uses the same vectory similarity retrieval for each subquery.

## Data
The data is directly from QUEST (https://github.com/google-research/language/tree/master/language/quest#examples).
- The documents that are embedded are: https://storage.googleapis.com/gresearch/quest/documents.zip
- `data/train_subset1.jsonl` is 20 randomly sampled queries from `train.jsonl` of QUEST.
- `data/train_subset2.jsonl` is 20 randomly sampled non-union queries from `train.jsonl` of QUEST.

## Retrieval
Retrieval is done with `semantic_retrieval/retrieve.py`.
- For decompose, we are retrieving titles only, and the code for this is written in `decompose/retrieve.py`
- For vector similarity, we are retrieving (title, chunk) tuples (`INCLUDE_CHUNKS = true`)

## Results:
|              | Retrieve (entire document) | Retrieve (first 512 tokens) | Decompose + Retrieve* (entire document)                         | Decompose + Retrieve* (first 512 tokens)                        |
|--------------|----------------------------|-----------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| Recall @ 20  | 0.0886                     | 0.1127                      | -                                                               |                                                                 |
| Recall @ 50  | 0.1663                     | 0.1593                      | 0.1560 (\|Pred\| = 61.10)                                       | 0.1617 (\|Pred\| = 60.70)                                       |
| Recall @ 100 | 0.2122                     | 0.2250                      | 0.2285 (\|Pred\| = 205.95) (k for subqueries increased in size) | 0.2157 (\|Pred\| = 209.30) (k for subqueries increased in size) |

## Data
