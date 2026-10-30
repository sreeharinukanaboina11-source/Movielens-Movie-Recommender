"""
Interactive Movie Recommender Application (CSE303 Week 10 Mini Project)
Topic: P01 - Movie Recommender (MovieLens latest-small)

Can be run as a CLI tool:
    python app.py --user 1 --n 10 --model hybrid
    python app.py --user 9999 --n 5   # Cold-start user demo

Or as a Streamlit web application (if streamlit is installed):
    streamlit run app.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_data_path():
    candidates = [
        'ml-latest-small',
        os.path.join('..', '..', '..', 'ml-latest-small'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-latest-small'),
        os.path.join('data', 'ml-latest-small')
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'ratings.csv')):
            return c
    raise FileNotFoundError("Could not find ml-latest-small folder. Please download it following data/README.md.")

class MovieRecommenderEngine:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = find_data_path()
        self.data_path = data_path
        self._load_and_train()
        
    def _load_and_train(self):
        # Load raw data
        ratings = pd.read_csv(os.path.join(self.data_path, 'ratings.csv')).rename(columns={'userId': 'user', 'movieId': 'item'})
        movies = pd.read_csv(os.path.join(self.data_path, 'movies.csv')).rename(columns={'movieId': 'item'})
        tags = pd.read_csv(os.path.join(self.data_path, 'tags.csv')).rename(columns={'movieId': 'item'})
        
        self.movies = movies
        self.movie_title_map = movies.set_index('item')['title'].to_dict()
        self.movie_genre_map = movies.set_index('item')['genres'].to_dict()
        
        self.users = sorted(ratings['user'].unique())
        self.items = sorted(ratings['item'].unique())
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {it: j for j, it in enumerate(self.items)}
        
        # Build training rating matrix
        self.R = ratings.pivot_table(index='user', columns='item', values='rating', aggfunc='mean').reindex(index=self.users, columns=self.items)
        self.global_mean = float(ratings['rating'].mean())
        self.user_means = self.R.mean(axis=1).fillna(self.global_mean).values
        
        self.R_vals = self.R.values
        self.R_mask = ~np.isnan(self.R_vals)
        self.seen_dict = ratings.groupby('user')['item'].apply(set).to_dict()
        
        # Mean-centered matrix and user similarities
        self.C_user = np.where(self.R_mask, self.R_vals - self.user_means[:, None], 0.0)
        self.sim_u = cosine_similarity(self.C_user)
        np.fill_diagonal(self.sim_u, 0.0)
        
        # Content model (TF-IDF on genres + tags)
        tag_text = tags.groupby('item')['tag'].apply(lambda x: ' '.join(x.astype(str).str.lower())).to_dict()
        genres_text = movies.set_index('item')['genres'].str.replace('|', ' ', regex=False).str.lower().to_dict()
        content_strings = [genres_text.get(it, '') + ' ' + tag_text.get(it, '') for it in self.items]
        
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        self.T_cb = self.tfidf.fit_transform(content_strings)
        
        # User profile representations
        self.profile_matrix = np.zeros((len(self.users), self.T_cb.shape[1]))
        for i in range(len(self.users)):
            rated_items = np.where(self.R_mask[i, :])[0]
            if len(rated_items) > 0:
                weights = self.C_user[i, rated_items]
                if np.all(weights == 0):
                    weights = self.R_vals[i, rated_items]
                self.profile_matrix[i] = np.asarray(self.T_cb[rated_items].T.dot(weights)).ravel()
        self.cb_scores_mat = cosine_similarity(self.profile_matrix, self.T_cb)
        
        # Popularity fallback table
        item_stats = ratings.groupby('item').agg(count=('rating', 'size'), mean_rating=('rating', 'mean'))
        self.popular_fallback = item_stats[item_stats['count'] >= 50].sort_values(by='mean_rating', ascending=False)

    def recommend(self, user_id, n=10, model='hybrid', k_nbrs=20):
        # Cold start user handling
        if user_id not in self.user2idx:
            top_pop = self.popular_fallback.head(n)
            results = []
            for rank, (it, row) in enumerate(top_pop.iterrows(), 1):
                results.append({
                    'Rank': rank,
                    'Movie ID': it,
                    'Title': self.movie_title_map.get(it, f"Movie {it}"),
                    'Genres': self.movie_genre_map.get(it, 'N/A'),
                    'Predicted Score': f"{row['mean_rating']:.2f}",
                    'Reason': f"Platform Top-Rated Hit ({int(row['count'])} ratings; new user fallback)"
                })
            return pd.DataFrame(results)
            
        uidx = self.user2idx[user_id]
        seen = self.seen_dict.get(user_id, set())
        
        # Determine dominant historical genres for explanation
        user_ratings = self.R.loc[user_id].dropna()
        if len(user_ratings) > 0:
            top_rated_item = user_ratings.sort_values(ascending=False).index[0]
            fav_genre = self.movie_genre_map.get(top_rated_item, 'Cinema').split('|')[0]
        else:
            fav_genre = 'Cinema'
            
        row_sim = self.sim_u[uidx]
        pos_nbrs = np.where(row_sim > 0)[0]
        if len(pos_nbrs) > k_nbrs:
            top_nbrs = pos_nbrs[np.argpartition(-row_sim[pos_nbrs], k_nbrs)[:k_nbrs]]
        else:
            top_nbrs = pos_nbrs
            
        cand_mask = np.any(self.R_mask[top_nbrs, :], axis=0) if len(top_nbrs) > 0 else np.ones(len(self.items), bool)
        cand_indices = np.where(cand_mask)[0]
        
        cand_scores = []
        u_mean = self.user_means[uidx]
        for itidx in cand_indices:
            it = self.items[itidx]
            if it in seen:
                continue
            raters = np.where(self.R_mask[top_nbrs, itidx])[0] if len(top_nbrs) > 0 else np.array([])
            if len(raters) > 0:
                actual_raters = top_nbrs[raters]
                sc = row_sim[actual_raters]
                pred_cf = u_mean + np.sum(sc * self.C_user[actual_raters, itidx]) / np.sum(sc)
            else:
                pred_cf = u_mean
                
            pred_cb = u_mean + self.cb_scores_mat[uidx, itidx] * 2.0
            
            if model == 'hybrid':
                score = 0.70 * pred_cf + 0.30 * pred_cb
            elif model == 'cf':
                score = pred_cf
            elif model == 'cb':
                score = pred_cb
            else:
                score = 0.70 * pred_cf + 0.30 * pred_cb
                
            cand_scores.append((it, score))
            
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = cand_scores[:n]
        
        results = []
        for rank, (it, score) in enumerate(top_n, 1):
            results.append({
                'Rank': rank,
                'Movie ID': it,
                'Title': self.movie_title_map.get(it, f"Movie {it}"),
                'Genres': self.movie_genre_map.get(it, 'N/A'),
                'Predicted Score': f"{score:.2f}",
                'Reason': f"Matches taste in {fav_genre} & co-rated by similar viewers"
            })
        return pd.DataFrame(results)

def main_cli():
    parser = argparse.ArgumentParser(description="Movie Recommender CLI Demo")
    parser.add_argument('--user', type=int, default=1, help="User ID to generate recommendations for (e.g. 1)")
    parser.add_argument('--n', type=int, default=10, help="Number of movies to recommend")
    parser.add_argument('--model', type=str, default='hybrid', choices=['hybrid', 'cf', 'cb'], help="Model algorithm")
    args = parser.parse_args()
    
    print(f"\nInitializing Movie Recommender Engine from {find_data_path()}...")
    engine = MovieRecommenderEngine()
    print(f"Engine ready. Generating top {args.n} recommendations for User {args.user} using {args.model.upper()} model:\n")
    recs = engine.recommend(args.user, n=args.n, model=args.model)
    print(recs.to_string(index=False))
    print()

def main_streamlit():
    import streamlit as st
    st.set_page_config(page_title="MovieLens Recommender", layout="wide", page_icon="🎬")
    st.title("🎬 MovieLens Recommender System (CSE303 Mini Project)")
    st.caption("P01 Foundation: Mean-Centred Collaborative Filtering, TF-IDF Content Model & Hybrid Fusion")
    
    @st.cache_resource
    def get_engine():
        return MovieRecommenderEngine()
        
    engine = get_engine()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        user_input = st.number_input("User ID", min_value=1, max_value=9999, value=1, step=1)
    with col2:
        model_choice = st.selectbox("Recommender Model", ['hybrid', 'cf', 'cb'], format_func=lambda x: x.upper())
    with col3:
        num_recs = st.slider("Number of Recommendations", min_value=3, max_value=20, value=10)
        
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Scoring movies..."):
            df = engine.recommend(user_id=user_input, n=num_recs, model=model_choice)
            st.dataframe(df, use_container_width=True)
            if user_input not in engine.user2idx:
                st.info("ℹ️ User ID was not found in the historical training set. Cold-start fallback was applied using top-rated movies with >= 50 ratings.")

if __name__ == '__main__':
    # If run through streamlit command, render streamlit UI
    if 'streamlit' in sys.modules or os.environ.get('STREAMLIT_RUN', False):
        try:
            main_streamlit()
        except Exception:
            main_cli()
    else:
        main_cli()
