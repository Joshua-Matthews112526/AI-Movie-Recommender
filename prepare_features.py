from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from recommender import load_movies


DATA_DIR = Path(__file__).parent / "data"

EMBEDDING_PATH = DATA_DIR / "movie_embeddings.npy"
INDEX_PATH = DATA_DIR / "movie_embedding_index.csv"

FEATURE_PATH = DATA_DIR / "movie_features.npy"
FEATURE_INDEX_PATH = DATA_DIR / "movie_feature_index.csv"


def create_features():
    movies = load_movies()

    embeddings = np.load(
        EMBEDDING_PATH
    )

    embedding_index = pd.read_csv(
        INDEX_PATH
    )

    # Make sure movie metadata is in exactly the same
    # order as the saved plot embeddings.
    feature_movies = embedding_index.merge(
        movies[
            [
                "id",
                "genres",
                "runtime",
                "year",
                "vote_average"
            ]
        ],
        on="id",
        how="left"
    )

    print(
        f"Building features for "
        f"{len(feature_movies)} movies..."
    )

    # ----------------------------
    # Genre features
    # ----------------------------

    genre_encoder = MultiLabelBinarizer()

    genre_features = (
        genre_encoder.fit_transform(
            feature_movies["genres"]
        )
    )

    print(
        f"Genre features: "
        f"{len(genre_encoder.classes_)}"
    )

    # ----------------------------
    # Numeric features
    # ----------------------------

    numeric_columns = [
        "runtime",
        "year",
        "vote_average"
    ]

    numeric_data = feature_movies[
        numeric_columns
    ].copy()

    # Handle any missing numeric values.
    for column in numeric_columns:
        numeric_data[column] = (
            numeric_data[column].fillna(
                numeric_data[column].median()
            )
        )

    # Runtime and year are much larger numbers
    # than our embedding values, so normalize them.
    scaler = StandardScaler()

    numeric_features = (
        scaler.fit_transform(
            numeric_data
        )
    )

    # ----------------------------
    # Combine everything
    # ----------------------------

    features = np.hstack(
        [
            embeddings,
            genre_features,
            numeric_features
        ]
    )

    np.save(
        FEATURE_PATH,
        features
    )

    feature_movies[
        ["id", "title"]
    ].to_csv(
        FEATURE_INDEX_PATH,
        index=False
    )

    print()
    print(
        "Feature matrix created:"
    )

    print(
        features.shape
    )

    print()
    print(
        f"Plot dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Genre dimensions: "
        f"{genre_features.shape[1]}"
    )

    print(
        f"Numeric dimensions: "
        f"{numeric_features.shape[1]}"
    )


if __name__ == "__main__":
    create_features()