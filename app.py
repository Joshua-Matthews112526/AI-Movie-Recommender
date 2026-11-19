import os

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    session,
    url_for
)

from recommender import (
    filter_movies,
    load_features,
    load_movies
)

from preference_model import (
    score_movies,
    select_pair,
    train_preference_model
)


app = Flask(__name__)

# Needed so Flask can store the user's
# comparison history in a session.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "development-secret-key"
)


# Load movie data once when Flask starts.
movies = load_movies()

features, feature_index = (
    load_features()
)


@app.route("/")
def home():
    """
    Starting page where the user enters
    optional filtering parameters.
    """

    return render_template(
        "index.html"
    )


@app.route("/compare")
def compare():
    """
    Show two movies and ask the user
    which one they would rather watch.
    """

    # If parameters were passed from the home page,
    # this is a brand new recommendation session.
    if (
        "max_runtime" in request.args
        or "genre" in request.args
    ):

        max_runtime = request.args.get(
            "max_runtime",
            type=int
        )

        genre = request.args.get(
            "genre",
            default=""
        ).strip()

        session["filters"] = {
            "max_runtime": max_runtime,
            "genre": genre
        }

        # Reset previous comparisons.
        session["history"] = []

    filters = session.get(
        "filters",
        {}
    )

    max_runtime = filters.get(
        "max_runtime"
    )

    genre = filters.get(
        "genre",
        ""
    )

    candidates = filter_movies(
        movies,
        max_runtime=max_runtime,
        genre=genre
    )

    if len(candidates) < 2:
        return render_template(
            "compare.html",
            error=(
                "Not enough movies matched "
                "those filters."
            )
        )

    history = session.get(
        "history",
        []
    )

    model = train_preference_model(
        history,
        features,
        feature_index
    )

    pair = select_pair(
        candidates["id"].tolist(),
        history,
        model,
        features,
        feature_index
    )

    if pair is None:
        return redirect(
            url_for("recommendation")
        )

    movie_a_id, movie_b_id = pair

    movie_a = candidates[
        candidates["id"] == movie_a_id
    ].iloc[0].to_dict()

    movie_b = candidates[
        candidates["id"] == movie_b_id
    ].iloc[0].to_dict()

    return render_template(
        "compare.html",
        movie_a=movie_a,
        movie_b=movie_b,
        comparison_count=len(history)
    )


@app.route(
    "/choose",
    methods=["POST"]
)
def choose():
    """
    Record one pairwise preference.
    """

    winner_id = int(
        request.form["winner_id"]
    )

    loser_id = int(
        request.form["loser_id"]
    )

    history = session.get(
        "history",
        []
    )

    history.append(
        {
            "winner_id": winner_id,
            "loser_id": loser_id
        }
    )

    session["history"] = history

    return redirect(
        url_for("compare")
    )


@app.route("/recommendation")
def recommendation():
    """
    Rank the candidate movies using everything
    the model has learned so far.
    """

    history = session.get(
        "history",
        []
    )

    if not history:
        return redirect(
            url_for("home")
        )

    filters = session.get(
        "filters",
        {}
    )

    candidates = filter_movies(
        movies,
        max_runtime=filters.get(
            "max_runtime"
        ),
        genre=filters.get(
            "genre",
            ""
        )
    )

    model = train_preference_model(
        history,
        features,
        feature_index
    )

    if model is None:
        return redirect(
            url_for("compare")
        )

    scores = score_movies(
        candidates["id"].tolist(),
        model,
        features,
        feature_index
    )

    if not scores:
        return redirect(
            url_for("home")
        )

    top_results = []

    for result in scores[:5]:

        movie_match = candidates[
            candidates["id"] == result["id"]
        ]

        if movie_match.empty:
            continue

        movie = movie_match.iloc[0].to_dict()

        movie["preference_score"] = result["score"]

        top_results.append(movie)


    return render_template(
        "recommendation.html",
        movies=top_results,
        comparisons=len(history)
    )


@app.route("/restart")
def restart():
    """
    Clear the current recommendation session.
    """

    session.clear()

    return redirect(
        url_for("home")
    )


if __name__ == "__main__":
    app.run(debug=True)