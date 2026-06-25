"""
Movie Recommender - Gradio Demo
=================================
HOW TO RUN
----------
1. Install dependencies:
       pip install gradio pandas scikit-learn
2. Run:
       python app.py
3. Gradio will print a local URL (and a public one if share=True) to open
   in your browser.
"""

import os
import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG - adjust these if your file/column names differ
# ---------------------------------------------------------------------------
RESULT_FOLDER = "data_engineering/result"
RATINGS_FILE = "ratings_dense.pkl"
MATRIX_FILE = "user_item_matrix.pkl"
MOVIES_FILE = "movies.pkl"      # optional, for titles
MOVIE_ID_COL = "movieId"
TITLE_COL = "title"


# ---------------------------------------------------------------------------
# Model classes (copied from Movie_recommondation_model.py).
# Only the reusable parts are kept here - the __main__ test block from the
# original file is NOT needed to run the actual app.
# ---------------------------------------------------------------------------
class PopularityBaselineRecommender:
    def __init__(self):
        self.top_movies = None
        self.global_mean = 0

    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        movie_stats = train_df.groupby('movieId').agg(
            rating_mean=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        self.top_movies = movie_stats.sort_values(by='rating_count', ascending=False)

    def predict_rating(self, user_id, movie_id):
        movie_idx = self.top_movies[self.top_movies['movieId'] == movie_id]
        if not movie_idx.empty:
            return movie_idx['rating_mean'].values[0]
        return self.global_mean

    def recommend_top_n(self, n=10):
        return self.top_movies['movieId'].head(n).tolist()


class AdvancedCollaborativeFiltering:
    def __init__(self):
        self.user_sim_df = None
        self.item_sim_df = None

    def calculate_cosine_similarity(self, user_item_matrix):
        matrix_filled = user_item_matrix.fillna(0)
        item_sim = cosine_similarity(matrix_filled.T)
        self.item_sim_df = pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        user_sim = cosine_similarity(matrix_filled)
        self.user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        return self.user_sim_df, self.item_sim_df

    def recommend_similar_and_popular(self, current_movie_id, baseline_model, top_k=20, final_n=5):
        """
        Finds movies with the most similar taste profile that are also
        the most popular in the market.
        """
        # Movie not in the similarity matrix (e.g. brand new movie) -> fall back to top hot movies
        if current_movie_id not in self.item_sim_df.index:
            return baseline_model.recommend_top_n(n=final_n)

        # Step 1: cosine similarity of current movie vs all others
        similar_scores = self.item_sim_df[current_movie_id].drop(current_movie_id)

        # Top K most similar movies
        top_similar_movies = similar_scores.sort_values(ascending=False).head(top_k).reset_index()
        top_similar_movies.columns = ['movieId', 'similarity_score']

        # Step 2: popularity stats from the baseline model
        movie_popularity = baseline_model.top_movies[['movieId', 'rating_count']]

        # Merge similarity + popularity
        merged_df = pd.merge(top_similar_movies, movie_popularity, on='movieId', how='inner')

        # Most-viewed movies first
        final_recommendations = merged_df.sort_values(by='rating_count', ascending=False)

        return final_recommendations['movieId'].head(final_n).tolist()


# ---------------------------------------------------------------------------
# Load data + train models ONCE at startup (not on every click)
# ---------------------------------------------------------------------------
print("Loading data...")
df_dense = pd.read_pickle(os.path.join(RESULT_FOLDER, RATINGS_FILE))
user_item_matrix = pd.read_pickle(os.path.join(RESULT_FOLDER, MATRIX_FILE))

# IMPORTANT FIX: user_item_matrix.pkl was produced via a CSV round-trip
# (data_transformation.py -> .csv -> transfer_to_pkl.py -> .pkl). CSV headers
# are always read back as strings, so the movieId COLUMN LABELS end up as
# '1', '2', '3'... (str) while movieId in ratings_dense.pkl is int64. Without
# this fix, current_movie_id (an int from the UI) never matches the matrix's
# string columns, so recommend_similar_and_popular() silently falls back to
# the popularity baseline for every single movie - the similarity logic
# never actually runs. Casting back to int here restores real behavior.
user_item_matrix.columns = user_item_matrix.columns.astype(int)

movies_df = None
movies_path = os.path.join(RESULT_FOLDER, MOVIES_FILE)
if os.path.exists(movies_path):
    try:
        candidate = pd.read_pickle(movies_path)
        if MOVIE_ID_COL in candidate.columns and TITLE_COL in candidate.columns:
            movies_df = candidate
            print(f"Loaded titles from {MOVIES_FILE}.")
        else:
            print(f"{MOVIES_FILE} found but missing expected columns "
                  f"('{MOVIE_ID_COL}', '{TITLE_COL}') - showing movie IDs instead.")
    except Exception as e:
        print(f"Could not read {MOVIES_FILE} ({e}) - showing movie IDs instead.")
else:
    print(f"{MOVIES_FILE} not found - showing movie IDs instead of titles.")

print("Training baseline (popularity) model...")
baseline = PopularityBaselineRecommender()
baseline.fit(df_dense)

print("Training advanced model (item-item cosine similarity)...")
advanced = AdvancedCollaborativeFiltering()
advanced.calculate_cosine_similarity(user_item_matrix)

print("Ready!")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_title(movie_id):
    if movies_df is not None:
        row = movies_df[movies_df[MOVIE_ID_COL] == movie_id]
        if not row.empty:
            return row[TITLE_COL].values[0]
    return f"Movie #{movie_id}"


def dropdown_label(movie_id):
    title = get_title(movie_id)
    if title == f"Movie #{movie_id}":
        return title
    return f"{title} (ID {movie_id})"


# Build dropdown choices: (label shown to user, actual value passed to the function)
if movies_df is not None:
    all_ids = sorted(movies_df[MOVIE_ID_COL].unique())
else:
    all_ids = sorted(user_item_matrix.columns)
MOVIE_CHOICES = [(dropdown_label(mid), int(mid)) for mid in all_ids]


# ---------------------------------------------------------------------------
# The function the UI calls
# ---------------------------------------------------------------------------
def recommend(selected_movie_id, top_k, final_n):
    if selected_movie_id is None:
        return pd.DataFrame(columns=["Movie ID", "Title"])

    movie_id = int(selected_movie_id)
    rec_ids = advanced.recommend_similar_and_popular(
        current_movie_id=movie_id,
        baseline_model=baseline,
        top_k=int(top_k),
        final_n=int(final_n),
    )
    return pd.DataFrame(
        [{"Movie ID": mid, "Title": get_title(mid)} for mid in rec_ids]
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Movie Recommender") as demo:
    gr.Markdown("# 🎬 Similar & Popular Movie Recommender")
    gr.Markdown(
        "Pick a movie you're currently watching. The system finds movies "
        "with a **similar taste profile**, then ranks those by **popularity** "
        "to give you its final picks."
    )

    movie_dropdown = gr.Dropdown(
        choices=MOVIE_CHOICES,
        label="Currently watching",
        value=MOVIE_CHOICES[0][1] if MOVIE_CHOICES else None,
        filterable=True,
    )

    with gr.Row():
        top_k_slider = gr.Slider(5, 50, value=20, step=1, label="Candidate pool size (top_k)")
        final_n_slider = gr.Slider(1, 20, value=5, step=1, label="Number of recommendations")

    recommend_btn = gr.Button("Get Recommendations", variant="primary")
    output_table = gr.Dataframe(headers=["Movie ID", "Title"], label="Recommended Movies")

    recommend_btn.click(
        fn=recommend,
        inputs=[movie_dropdown, top_k_slider, final_n_slider],
        outputs=output_table,
    )

if __name__ == "__main__":
    demo.launch()
#demo.launch(share=True) change demo.launch() to this when use on gg colab, use this to install needed libraries!pip install gradio pandas scikit-learn -q
