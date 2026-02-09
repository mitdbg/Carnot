import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/papers_with_related_works.csv")
    parser.add_argument("--output-dir", type=str, default="carnot_benchmark_corpus")
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    df = pd.read_csv(dataset_path)
    if args.max_papers:
        df = df.head(args.max_papers)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        content = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
        arxiv_id = str(row["arxiv_id"]).replace("/", "_")
        filename = f"{idx}_{arxiv_id}.txt"
        (output_dir / filename).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
