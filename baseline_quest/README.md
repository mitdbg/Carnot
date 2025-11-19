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
    - Generate decompositions with `decompose/generate_decompositions.py`

2. Retrieval (@ k)
    - Retrieve with query / subquery (vector similarity)
        - If indexed entired documented, retrieves the top 200 most likely chunks, then maps them to the document
        - If indexed first 512 tokens only, retrieves the top k most likely chunks


Results:
|              | Retrieve (entire document) | Retrieve (first 512 tokens) | Decompose + Retrieve* (entire document)                         | Decompose + Retrieve* (first 512 tokens)                        |
|--------------|----------------------------|-----------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| Recall @ 20  | 0.0886                     | 0.1127                      | -                                                               |                                                                 |
| Recall @ 50  | 0.1663                     | 0.1593                      | 0.1560 (\|Pred\| = 61.10)                                       | 0.1617 (\|Pred\| = 60.70)                                       |
| Recall @ 100 | 0.2122                     | 0.2250                      | 0.2285 (\|Pred\| = 205.95) (k for subqueries increased in size) | 0.2157 (\|Pred\| = 209.30) (k for subqueries increased in size) |