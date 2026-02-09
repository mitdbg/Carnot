import re
import sys
from pathlib import Path

import pandas as pd

CARNOT_PATH = Path(__file__).resolve().parent.parent.parent / "Carnot" / "src"
if CARNOT_PATH.exists() and str(CARNOT_PATH) not in sys.path:
    sys.path.insert(0, str(CARNOT_PATH))

try:
    import carnot
except ImportError:
    print("Carnot not found. Add to PYTHONPATH or activate Carnot venv.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "papers_with_related_works.csv"
CORPUS_DIR = PROJECT_ROOT / "carnot_benchmark_corpus"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "baselines_results" / "carnot_deep_research"
NUM_PAPERS = 5

RELATED_WORKS_QUERY_TEMPLATE = """Write a Related Works section for an academic paper given the paper's abstract.

Here is the paper abstract:
{abstract}

Guidelines:
- Write a cohesive related work section as though it is part of an academic research paper at a top conference.
- Cite papers using markdown links: [Author et al. Year](http://arxiv.org/abs/arxiv_id)
- Only cite papers that appear in the provided documents. Use their arXiv IDs in the URLs.
- Organize by themes or chronology. Include 10-25 citations."""


def create_corpus_if_needed():
    if CORPUS_DIR.exists() and list(CORPUS_DIR.glob("*.txt")):
        return
    import subprocess

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "create_carnot_corpus.py")],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def extract_answer_from_carnot_output(output) -> str:
    df = output.to_df()
    for col in df.columns:
        if col.startswith("result-") or col == "context":
            vals = df[col].dropna()
            for v in vals:
                if isinstance(v, str) and len(v) > 200:
                    return v
    best = ""
    for col in df.columns:
        for v in df[col].dropna():
            s = str(v).strip()
            if len(s) > len(best):
                best = s
    return best


def extract_arxiv_citations(text: str) -> list[str]:
    pattern = r"arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)"
    return list(dict.fromkeys(re.findall(pattern, text, re.IGNORECASE)))


def build_paper_csv(arxiv_ids: list[str], paper_map: dict) -> pd.DataFrame:
    rows = []
    seen = set()
    for aid in arxiv_ids:
        if aid in seen:
            continue
        seen.add(aid)
        base = aid.replace("v1", "").replace("v2", "")
        info = (
            paper_map.get(aid)
            or paper_map.get(aid + "v1")
            or paper_map.get(base)
            or paper_map.get(base + "v1")
        )
        if info:
            rows.append(
                {
                    "id": info["arxiv_id"],
                    "title": info["title"],
                    "snippet": info.get("abstract", ""),
                }
            )
        else:
            rows.append({"id": aid, "title": f"ArXiv {aid}", "snippet": ""})
    return pd.DataFrame(rows, columns=["id", "title", "snippet"])


def run_single_paper(file_id: int, df: pd.DataFrame, paper_map: dict) -> tuple[str, pd.DataFrame]:
    row = df.iloc[file_id]
    abstract = row["abstract"]
    title = row["title"]

    query = RELATED_WORKS_QUERY_TEMPLATE.format(abstract=abstract)

    ctx = carnot.TextFileContext(
        str(CORPUS_DIR.resolve()),
        id=f"bench-paper-{file_id}",
        description=f"Academic papers corpus for Related Works. Target: {title[:100]}...",
    )
    compute_ctx = ctx.compute(query)
    output = compute_ctx.run(config=carnot.QueryProcessorConfig(progress=True))

    intro_text = extract_answer_from_carnot_output(output)
    if not intro_text.strip():
        intro_text = f"## Related Works\n\n(No content generated for paper {file_id})"

    if "## Related Works" not in intro_text[:50] and "Related Work" not in intro_text[:50]:
        intro_text = "## Related Works\n\n" + intro_text

    arxiv_ids = extract_arxiv_citations(intro_text)
    paper_df = build_paper_csv(arxiv_ids, paper_map)

    return intro_text, paper_df


def main():
    create_corpus_if_needed()

    df = pd.read_csv(DATASET_PATH)
    paper_map = {}
    for _, row in df.iterrows():
        aid = str(row["arxiv_id"]).strip()
        paper_map[aid] = {
            "arxiv_id": aid,
            "title": str(row["title"]).strip(),
            "abstract": str(row["abstract"]).strip() if pd.notna(row["abstract"]) else "",
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_id in range(min(NUM_PAPERS, len(df))):
        intro_text, paper_df = run_single_paper(file_id, df, paper_map)
        out_folder = OUTPUT_DIR / str(file_id)
        out_folder.mkdir(parents=True, exist_ok=True)
        (out_folder / "intro.md").write_text(intro_text, encoding="utf-8")
        paper_df.to_csv(out_folder / "paper.csv", index=False)


if __name__ == "__main__":
    main()
