import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict

# Load the dataset
data = pd.read_csv('../data/emotions_cleaned.csv')

# Ensure no NaNs in 'text' column
data['text'] = data['text'].fillna('')

# Initialize the TF-IDF Vectorizer with stop words removed
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the text data
tfidf_sparse_matrix = tfidf_vectorizer.fit_transform(data['text'])

# Map numerical labels to emotion names
label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['label'] = data['label'].map(label_map)

labels = data['label'].values

# Define words to exclude from top terms
excluded_words = {'really', 'little'}

# Function to calculate top TF-IDF terms
def get_top_tfidf_terms_sparse(matrix, labels, label, n=20):
    label_indices = labels == label
    label_matrix = matrix[label_indices]
    mean_tfidf = label_matrix.mean(axis=0)
    mean_tfidf = np.array(mean_tfidf).flatten()
    top_terms = [(tfidf_vectorizer.get_feature_names_out()[i], mean_tfidf[i]) for i in mean_tfidf.argsort()[-n:][::-1]]
    return top_terms

# Compute top terms for initial analysis
top_tfidf_terms_sparse = {}
for label in np.unique(labels):
    top_tfidf_terms_sparse[label] = get_top_tfidf_terms_sparse(tfidf_sparse_matrix, labels, label)

# Identify common terms across emotions based on cumulative TF-IDF score
term_scores_across_emotions = defaultdict(float)
term_count_across_emotions = defaultdict(int)

for emotion, terms in top_tfidf_terms_sparse.items():
    for term, score in terms:
        term_scores_across_emotions[term] += score
        term_count_across_emotions[term] += 1

# Set the threshold
threshold = 0.06

# Determine common high-value terms across all emotions
common_high_value_terms = {term for term, count in term_count_across_emotions.items() if count == len(label_map) and term_scores_across_emotions[term] > threshold}

# Function to get top unique TF-IDF terms
def get_top_unique_tfidf_terms_sparse(matrix, labels, label, n=5):
    label_indices = labels == label
    label_matrix = matrix[label_indices]
    mean_tfidf = label_matrix.mean(axis=0)
    mean_tfidf = np.array(mean_tfidf).flatten()

    top_terms = []
    for idx in mean_tfidf.argsort()[-n * 3:][::-1]:  # Looking deeper into the sorted list
        term = tfidf_vectorizer.get_feature_names_out()[idx]
        if term not in common_high_value_terms and term not in excluded_words:
            top_terms.append((term, mean_tfidf[idx]))
        if len(top_terms) == n:
            break
    return top_terms

# Compute top unique terms for each emotion
top_unique_tfidf_terms_sparse = {}
for label in np.unique(labels):
    top_unique_tfidf_terms_sparse[label] = get_top_unique_tfidf_terms_sparse(tfidf_sparse_matrix, labels, label)

# Display the results
for emotion, terms in top_unique_tfidf_terms_sparse.items():
    print(f"Top 5 unique TF-IDF terms for {emotion}:")
    for term, score in terms:
        print(f"{term}: {score}")
    print()
