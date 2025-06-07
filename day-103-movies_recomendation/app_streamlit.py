import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import gdown
import io

# Load environment variable (API key)
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# URL Google Drive untuk file
MOVIES_URL = os.getenv('MOVIES_URL')  # Ganti dengan ID file movies.pkl
SIMILARITY_URL = os.getenv('SIMILARITY_URL')  # Ganti dengan ID file similarity.pkl

# Fungsi untuk mengunduh dan memuat file dari Google Drive
def load_from_drive(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return pickle.load(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error loading file from {url}: {e}")
        return None


# Load data dari Google Drive
movies_df = pd.DataFrame(load_from_drive(MOVIES_URL))
similarity = load_from_drive(SIMILARITY_URL)
if similarity is None:
    print("Failed to load similarity.pkl from Google Drive")
else:
    print(f"Loaded similarity.pkl with shape {similarity.shape if hasattr(similarity, 'shape') else 'unknown'}")

# Fetch poster and Google search link
def fetch_poster_and_link(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    if response.status_code != 200:
        return "https://via.placeholder.com/300x450?text=Error", "#"
    data = response.json()
    if data["results"]:
        poster_path = data["results"][0].get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
    else:
        poster_url = "https://via.placeholder.com/300x450?text=No+Image"
    search_link = f"https://www.google.com/search?q={title.replace(' ', '+')}+movie"
    return poster_url, search_link

# Main recommendation function
def recommend(movie):
    movie = movie.casefold()
    if movie not in movies_df['title'].str.casefold().values:
        return None
    idx = movies_df[movies_df['title'].str.casefold() == movie].index[0]
    distances = similarity[idx]
    top_matches = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]
    
    titles, posters, links = [], [], []
    for i in top_matches:
        title = movies_df.iloc[i[0]].title
        poster, link = fetch_poster_and_link(title)
        titles.append(title)
        posters.append(poster)
        links.append(link)
    return titles, posters, links

# Streamlit UI
st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Search for a movie",
    sorted(movies_df['title']),
    index=None,
    placeholder="Type to search..."
)

if st.button("Recommend") and selected_movie_name:
    result = recommend(selected_movie_name)
    if result:
        names, posters, links = result
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"**[{names[idx]}]({links[idx]})**", unsafe_allow_html=True)
                st.image(posters[idx], use_container_width=True)
    else:
        st.error("Movie not found. Please try another title.")

# to running write streamlit run new_app.py in the terminal