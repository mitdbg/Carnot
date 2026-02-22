import os

import pandas as pd
import pytest


@pytest.fixture
def movie_reviews_data():
    movies_df = pd.read_csv("tests/pytest/data/movie-reviews/4/rotten_tomatoes_movies.csv")
    reviews_df = pd.read_csv("tests/pytest/data/movie-reviews/4/rotten_tomatoes_movie_reviews.csv")

    movie_ids = ["inception", "volver", "mean_girls"]
    filtered_movies_df = movies_df[movies_df["id"].isin(movie_ids)]
    filtered_reviews_df = reviews_df[reviews_df["id"].isin(movie_ids)]

    return filtered_movies_df, filtered_reviews_df


@pytest.fixture
def research_papers_data():
    papers = []
    for paper in os.listdir("tests/pytest/data/papers"):
        with open(f"tests/pytest/data/papers/{paper}") as f:
            contents = f.read()
            papers.append({"contents": contents})

    return papers

@pytest.fixture
def enron_emails_data():
    emails = []
    for email in os.listdir("tests/pytest/data/enron-eval-medium"):
        with open(f"tests/pytest/data/enron-eval-medium/{email}") as f:
            contents = f.read()
            emails.append({"contents": contents})
    return emails


@pytest.fixture
def enron_data_items():
    """DataItems with paths to Enron email files (for Flat/Hierarchical indices)."""
    from pathlib import Path

    from carnot.data.item import DataItem

    enron_dir = Path(__file__).resolve().parent.parent / "data" / "enron-eval-medium"
    if not enron_dir.exists():
        pytest.skip(f"Enron data dir not found: {enron_dir}")
    return [DataItem(path=str(p.absolute())) for p in sorted(enron_dir.glob("*.txt"))]
