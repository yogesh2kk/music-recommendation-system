import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('tcc_ceds_music-1.csv')

# Clean column names: remove leading/trailing spaces, replace spaces with underscores
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')

# Select features for recommendation
numeric_features = ['len', 'dating', 'violence', 'world/life', 'shake_the_audience', 'family/gospel',
                    'romantic', 'communication', 'obscene', 'music', 'movement/places', 'light/visual_perceptions',
                    'family/spiritual', 'like/girls', 'sadness', 'feelings', 'danceability',
                    'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy', 'age']

# Handle categorical variables: 'genre' and 'topic'
df = pd.get_dummies(df, columns=['genre', 'topic'])

# Impute and normalize numeric features
imputer = SimpleImputer(strategy='mean')
df[numeric_features] = imputer.fit_transform(df[numeric_features])

scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Select a range of possible n_components values
n_components_range = range(5, 23)

# List to store the explained variances for each n_components
explained_variances = []

# Loop through n_components
for n_components in n_components_range:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    features_transformed = svd.fit_transform(df[numeric_features])
    explained_variance = svd.explained_variance_ratio_.sum()
    explained_variances.append(explained_variance)

# Plot explained variance as a function of n_components
plt.plot(n_components_range, explained_variances)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.title('Explained variance vs Number of components')
plt.show()

# Select the optimal n_components: the smallest number that explains most variance
optimal_n_components = n_components_range[np.argmax(explained_variances)]

# Re-run SVD with the optimal n_components
svd = TruncatedSVD(n_components=optimal_n_components, random_state=42)
features_transformed = svd.fit_transform(df[numeric_features])

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(features_transformed)

def recommend_songs(df, song_index, cosine_sim_matrix, n_recommendations=5):
    # Get the pairwise similarity scores for all songs with that song
    cosine_sim_scores = list(enumerate(cosine_sim_matrix[song_index]))

    # Sort the songs based on the cosine similarity scores
    cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores for n most similar songs
    cosine_sim_scores = cosine_sim_scores[1:n_recommendations+1]

    # Get the song indices
    song_indices = [i[0] for i in cosine_sim_scores]

    return df.iloc[song_indices]

# Test
song_index = 24000
recommendations = recommend_songs(df, song_index, cosine_sim_matrix)
print(recommendations[['artist_name', 'track_name']])

# Create t-SNE object
tsne = TSNE(n_components=2)

# Fit and transform data to 2D
data_2d = tsne.fit_transform(features_transformed)

# Get song indices for the most similar songs
song_indices = recommendations.index.values

# Plot songs
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)

# Highlight the song and its recommendations
plt.scatter(data_2d[song_indices, 0], data_2d[song_indices, 1], color='red')

# Additional highlighting for the selected song
plt.scatter(data_2d[song_index, 0], data_2d[song_index, 1], color='blue')

plt.show()
