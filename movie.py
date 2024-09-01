import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Replace with your TMDb API key
TMDB_API_KEY = 'xxx'
def get_user_watch_history_tmdb(user_id):
    url = f"https://api.themoviedb.org/3/account/{user_id}/watchlist/movies"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US',
        'sort_by': 'created_at.asc'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        raise Exception("Failed to fetch user watch history")

# Example usage
user_id = 'xxx'  # Replace with a valid user ID
user_history = get_user_watch_history_tmdb(user_id)
def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch movie details")

# Fetch details for all watched movies
movies_data = [get_movie_details(movie['id']) for movie in user_history]

# Convert the list to a DataFrame
movies_df = pd.DataFrame(movies_data)
movies_df['genres'] = movies_df['genres'].apply(lambda x: " ".join([genre['name'] for genre in x]))
def analyze_watch_patterns(movies_df):
    # Analyzing Genre Preferences
    genre_series = movies_df['genres'].str.split(expand=True).stack()
    genre_counts = genre_series.value_counts()
    
    # Analyzing Binge-Watching (assumes `user_history` has `watch_time` field)
    movies_df['watch_time'] = pd.to_datetime(movies_df['release_date'])  # Use release_date as a proxy
    movies_df['binge_watch'] = movies_df['watch_time'].diff().dt.days <= 1
    
    binge_counts = movies_df['binge_watch'].value_counts()
    
    return genre_counts, binge_counts

# Example usage
genre_counts, binge_counts = analyze_watch_patterns(movies_df)

# Display genre preferences
print("Genre Preferences:")
print(genre_counts)

# Display binge-watching statistics
print("\nBinge-Watching Stats:")
print(binge_counts)
def genre_based_recommendation(movies_df, top_n=5):
    # Vectorize the genres
    count_vectorizer = CountVectorizer(stop_words='english')
    genre_matrix = count_vectorizer.fit_transform(movies_df['genres'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    
    # Recommend top N movies similar to the user's watched list
    sim_scores = cosine_sim.sum(axis=0)  # Sum similarities across all movies
    movie_indices = sim_scores.argsort()[-top_n:][::-1]  # Get top N indices
    
    recommendations = movies_df.iloc[movie_indices]
    return recommendations['title'].values

# Example usage
recommendations = genre_based_recommendation(movies_df)
print("Genre-based Recommendations:", recommendations)
def plot_watch_patterns(genre_counts, binge_counts):
    # Plot Genre Preferences
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.title('Genre Preferences')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # Plot Binge-Watching Stats
    plt.figure(figsize=(6, 4))
    sns.barplot(x=binge_counts.index, y=binge_counts.values, palette='coolwarm')
    plt.title('Binge-Watching Analysis')
    plt.xlabel('Binge-Watched')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

# Example usage
plot_watch_patterns(genre_counts, binge_counts)
def run_full_analysis(user_id):
    # Step 1: Get User Watch History
    user_history = get_user_watch_history_tmdb(user_id)
    
    # Step 2: Fetch Movie Details
    movies_data = [get_movie_details(movie['id']) for movie in user_history]
    movies_df = pd.DataFrame(movies_data)
    movies_df['genres'] = movies_df['genres'].apply(lambda x: " ".join([genre['name'] for genre in x]))

    # Step 3: Analyze Watch Patterns
    genre_counts, binge_counts = analyze_watch_patterns(movies_df)
    
    # Step 4: Recommend Movies
    recommendations = genre_based_recommendation(movies_df)
    print("Genre-based Recommendations:", recommendations)
    
    # Step 5: Plot Watch Patterns
    plot_watch_patterns(genre_counts, binge_counts)

# Run the full analysis
run_full_analysis(user_id)
