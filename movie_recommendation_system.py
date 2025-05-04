import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
# Dataset sourced from: https://github.com/devensinghbhagtani/Bollywood-Movie-Dataset/blob/main/IMDB-Movie-Dataset(2023-1951).csv
df = pd.read_csv('IMDB-Movie-Dataset(2023-1951).csv')

# Fill missing values
for col in ['overview', 'genre', 'cast', 'director']:
    df[col] = df[col].fillna('')

# Create separate columns for different content types
df['story_theme'] = (df['overview'] + ' ') * 3 + (df['genre'] + ' ') * 2
df['people_info'] = df['cast'] + ' ' + df['director']

# Combine both into one column for similarity search
df['combined_features'] = df['story_theme'] + ' ' + df['people_info']

# TF-IDF Vectorizer focused more on story/theme
vectorizer = TfidfVectorizer(stop_words='english', max_features=7000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping for movie names
indices = pd.Series(df.index, index=df['movie_name']).drop_duplicates()

# Recommendation function
def get_recommendations(movie_name, cosine_sim=cosine_sim):
    if movie_name not in indices:
        return f"Movie '{movie_name}' not found in dataset."

    idx = indices[movie_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 (excluding itself)
    movie_indices = [i[0] for i in sim_scores]
    return df[['movie_name']].iloc[movie_indices]  # Show more info for context

# Test it
recommended = get_recommendations('Add Movie')
print("Top Recommendations:\n", recommended)
