import pandas as pd
import numpy as np
import json
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nltk.download('punkt')
stemmer = PorterStemmer()

def stem(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def preprocess():
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies = movies.merge(credits, on='title')
    
    df = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

    def extract_names(obj_str, top_n=None):
        try:
            data = json.loads(obj_str)
            names = [d['name'] for d in data]
            return names[:top_n] if top_n else names
        except:
            return []

    def extract_director(obj_str):
        try:
            data = json.loads(obj_str)
            for d in data:
                if d['job'] == 'Director':
                    return [d['name']]
            return []
        except:
            return []

    df['genres'] = df['genres'].apply(extract_names)
    df['keywords'] = df['keywords'].apply(extract_names)
    df['cast'] = df['cast'].apply(lambda x: extract_names(x, 3))
    df['crew'] = df['crew'].apply(extract_director)

    df['overview'] = df['overview'].apply(lambda x: x.split())
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x))
    df['tags'] = df['tags'].apply(lambda x: x.lower())
    df['tags'] = df['tags'].apply(stem)

    new_df = df[['movie_id', 'title', 'tags']]

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    # Save to disk
    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

if __name__ == "__main__":
    preprocess()
    print("Data berhasil diproses dan disimpan.")
