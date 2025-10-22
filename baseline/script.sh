python3 evaluate_queries_baseline.py /orcd/home/002/joycequ/quest_data/train.jsonl 61 237 339 390 392 400 424 517 595 642 655 786 817 857 944 990 1004 1099 1111 1154

python3 generate_predictions.py /orcd/home/002/joycequ/quest_data/train.jsonl predictions_subset.jsonl 61 237 339 390 392 400 424 517 595 642 655 786 817 857 944 990 1004 1099 1111 1154

python3 compute_metrics.py --gold /orcd/home/002/joycequ/quest_data/train.jsonl --pred predictions_subset.jsonl

python3 entry_extractor.py /Users/joycequ/Documents/UROP/quest_data/train.jsonl /Users/joycequ/Documents/UROP/carnot-orcd/train_subset.jsonl --indices 61 237 339 390 392 400 424 517 595 642 655 786 817 857 944 990 1004 1099 1111 1154

python3 analyze_retriever.py --gold train_subset.jsonl --pred pred_unranked.jsonl