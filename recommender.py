from pathlib import Path
import ast

import pandas as pd

import numpy as np


DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "tmdb_5000_movies.csv"
)

EMBEDDING_PATH = (
    Path(__file__).parent
    / "data"
    / "movie_embeddings.npy"
)

INDEX_PATH = (
    Path(__file__).parent
    / "data"
    / "movie_embedding_index.csv"
)


def parse_names(value):
    """
    Convert TMDB JSON-style columns into a simple list of names.

    Example:
    [{"id": 28, "name": "Action"},
     {"id": 12, "name": "Adventure"}]

    becomes:

    ["Action", "Adventure"]
    """

    try:
        items = ast.literal_eval(value)

        return [
            item["name"]
            for item in items
            if "name" in item
        ]

    except (ValueError, SyntaxError, TypeError):
        return []


def load_movies():
    """
    Load and clean the TMDB movie dataset.
    """

    movies = pd.read_csv(DATA_PATH)

    # Keep only the columns useful to our project.
    movies = movies[
        [
            "id",
            "title",
            "genres",
            "runtime",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
            "overview",
        ]
    ].copy()

    # Convert genres from TMDB JSON text into Python lists.
    movies["genres"] = movies["genres"].apply(parse_names)

    # Convert release date into a year.
    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce"
    )

    movies["year"] = movies[
        "release_date"
    ].dt.year

    # Remove movies that are missing important information.
    movies = movies.dropna(
        subset=[
            "title",
            "runtime",
            "overview",
        ]
    )

    # Remove movies with invalid runtimes.
    movies = movies[
        movies["runtime"] > 0
    ]

    # Make runtime an integer.
    movies["runtime"] = movies[
        "runtime"
    ].astype(int)

    return movies


def filter_movies(
    movies,
    max_runtime=None,
    genre=None
):
    """
    Apply hard user constraints before ML ranking.
    """

    filtered = movies.copy()

    if max_runtime is not None:
        filtered = filtered[
            filtered["runtime"] <= max_runtime
        ]

    if genre:
        filtered = filtered[
            filtered["genres"].apply(
                lambda genres:
                genre.lower()
                in [
                    g.lower()
                    for g in genres
                ]
            )
        ]

    # Temporary ranking.
    # This will eventually be replaced with ML scoring.
    filtered = filtered.sort_values(
        by="vote_average",
        ascending=False
    )

    return filtered

def load_embeddings():
    """
    Load the saved movie embeddings and their index.
    """

    embeddings = np.load(EMBEDDING_PATH)

    embedding_index = pd.read_csv(INDEX_PATH)

    return embeddings, embedding_index


def find_similar_movies(
    movie_title,
    movies,
    embeddings,
    embedding_index,
    num_results=5
):
    """
    Find movies with plot embeddings most similar
    to the selected movie.
    """

    matches = embedding_index[
        embedding_index["title"].str.lower()
        == movie_title.lower()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    selected_embedding = embeddings[movie_index]

    # Embeddings were normalized when generated,
    # so the dot product acts as cosine similarity.
    similarities = embeddings @ selected_embedding

    # Sort from highest similarity to lowest.
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices:

        # Skip the movie itself.
        if index == movie_index:
            continue

        movie_id = embedding_index.iloc[index]["id"]

        movie_match = movies[
            movies["id"] == movie_id
        ]

        if movie_match.empty:
            continue

        movie = movie_match.iloc[0].to_dict()

        movie["similarity"] = float(
            similarities[index]
        )

        results.append(movie)

        if len(results) >= num_results:
            break

    return results