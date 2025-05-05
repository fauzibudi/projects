import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

def fetch_poster_and_link(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response["results"]:
        poster_path = response["results"][0].get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            poster_url = "https://via.placeholder.com/300x450?text=No+Image"
    else:
        poster_url = "https://via.placeholder.com/300x450?text=No+Image"
    google_link = f"https://www.google.com/search?q={title.replace(' ', '+')}+movie"
    return poster_url, google_link

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return None
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_titles = []
    poster_urls = []
    search_links = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster_url, search_url = fetch_poster_and_link(title)
        recommended_titles.append(title)
        poster_urls.append(poster_url)
        search_links.append(search_url)
    return recommended_titles, poster_urls, search_links

# Streamlit UI
st.title("Movie Recommender System")

movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    result = recommend(movie_name)
    if result:
        names, posters, links = result
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"**[{names[idx]}]({links[idx]})**", unsafe_allow_html=True)
                st.image(posters[idx], use_container_width=True)
    else:
        st.error("Movie not found. Please try another title.")
