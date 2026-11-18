import numpy as np
from sklearn.linear_model import LogisticRegression


def create_id_to_index(embedding_index):
    """
    Maps each TMDB movie ID to its row
    in the embedding matrix.
    """
    return {
        int(movie_id): index
        for index, movie_id
        in enumerate(embedding_index["id"])
    }


def train_preference_model(
    history,
    embeddings,
    embedding_index
):
    """
    Train a pairwise preference model.

    history should look like:

    [
        {
            "winner_id": 123,
            "loser_id": 456
        },
        ...
    ]
    """

    if not history:
        return None

    id_to_index = create_id_to_index(
        embedding_index
    )

    training_data = []
    labels = []

    for comparison in history:
        winner_id = comparison["winner_id"]
        loser_id = comparison["loser_id"]

        if (
            winner_id not in id_to_index
            or loser_id not in id_to_index
        ):
            continue

        winner_embedding = embeddings[
            id_to_index[winner_id]
        ]

        loser_embedding = embeddings[
            id_to_index[loser_id]
        ]

        # Difference between the movie the user preferred
        # and the movie they rejected.
        difference = (
            winner_embedding
            - loser_embedding
        )

        # Winner > Loser
        training_data.append(difference)
        labels.append(1)

        # Add the reverse example as well.
        # Loser < Winner
        training_data.append(-difference)
        labels.append(0)

    if len(training_data) < 2:
        return None

    X = np.array(training_data)
    y = np.array(labels)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    return model


def score_movies(
    movie_ids,
    model,
    embeddings,
    embedding_index
):
    """
    Give movies a preference score based on
    what the model has learned.
    """

    id_to_index = create_id_to_index(
        embedding_index
    )

    results = []

    for movie_id in movie_ids:

        movie_id = int(movie_id)

        if movie_id not in id_to_index:
            continue

        embedding = embeddings[
            id_to_index[movie_id]
        ]

        # The learned logistic-regression weights
        # act as the user's current preference vector.
        score = np.dot(
            embedding,
            model.coef_[0]
        )

        results.append(
            {
                "id": movie_id,
                "score": float(score)
            }
        )

    return sorted(
        results,
        key=lambda movie: movie["score"],
        reverse=True
    )

import random


def select_pair(
    movie_ids,
    history,
    model,
    embeddings,
    embedding_index
):
    """
    Select two movies for the user to compare.

    Before the model has learned anything:
        Choose an unseen random pair.

    After training:
        Choose two highly ranked movies with similar
        scores. A close decision gives the model
        useful information about the user's preferences.
    """

    movie_ids = [
        int(movie_id)
        for movie_id in movie_ids
    ]

    if len(movie_ids) < 2:
        return None

    # Keep track of pairs the user has already seen.
    compared_pairs = {
        frozenset(
            [
                int(item["winner_id"]),
                int(item["loser_id"])
            ]
        )
        for item in history
    }

    # If we do not have a trained model yet,
    # choose a random unseen pair.
    if model is None:

        available_pairs = []

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):

                pair = frozenset(
                    [movie_ids[i], movie_ids[j]]
                )

                if pair not in compared_pairs:
                    available_pairs.append(
                        (
                            movie_ids[i],
                            movie_ids[j]
                        )
                    )

        if not available_pairs:
            return None

        return random.choice(
            available_pairs
        )

    # Score every candidate movie.
    ranked = score_movies(
        movie_ids,
        model,
        embeddings,
        embedding_index
    )

    # Focus on the movies the model currently
    # thinks are strongest candidates.
    top_movies = ranked[:25]

    best_pair = None
    smallest_difference = float("inf")

    for i in range(len(top_movies)):
        for j in range(i + 1, len(top_movies)):

            movie_a = top_movies[i]
            movie_b = top_movies[j]

            pair = frozenset(
                [
                    movie_a["id"],
                    movie_b["id"]
                ]
            )

            if pair in compared_pairs:
                continue

            difference = abs(
                movie_a["score"]
                - movie_b["score"]
            )

            if difference < smallest_difference:

                smallest_difference = difference

                best_pair = (
                    movie_a["id"],
                    movie_b["id"]
                )

    # If every top pair has already been compared,
    # fall back to a random unseen pair.
    if best_pair is None:

        remaining_pairs = []

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):

                pair = frozenset(
                    [movie_ids[i], movie_ids[j]]
                )

                if pair not in compared_pairs:
                    remaining_pairs.append(
                        (
                            movie_ids[i],
                            movie_ids[j]
                        )
                    )

        if not remaining_pairs:
            return None

        return random.choice(
            remaining_pairs
        )

    return best_pair