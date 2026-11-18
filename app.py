from flask import Flask, render_template, request

from recommender import load_movies, filter_movies

app = Flask(__name__)

movies = load_movies()


@app.route("/")
def home():
    max_runtime = request.args.get(
        "max_runtime",
        type=int
    )

    genre = request.args.get(
        "genre",
        default=""
    ).strip()

    results = filter_movies(
        movies,
        max_runtime=max_runtime,
        genre=genre
    )

    return render_template(
        "index.html",
        movies=results.to_dict(orient="records"),
        max_runtime=max_runtime or "",
        genre=genre
    )


if __name__ == "__main__":
    app.run(debug=True)