from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

data = pd.read_csv('../data/emotions.csv')

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

tfidf_sparse_matrix = tfidf_vectorizer.fit_transform(data['text'])

def get_top_tfidf_terms_sparse(matrix, labels, label, n=5):
    label_indices = labels == label
    label_matrix = matrix[label_indices]
    mean_tfidf = label_matrix.mean(axis=0)
    mean_tfidf = np.array(mean_tfidf).flatten()
    top_indices = mean_tfidf.argsort()[-n:][::-1]
    top_terms = [(tfidf_vectorizer.get_feature_names_out()[i], mean_tfidf[i]) for i in top_indices]
    return top_terms

top_tfidf_terms_sparse = {}

labels = data['label'].values

label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

for label in np.unique(labels):
    top_tfidf_terms_sparse[label_map[label]] = get_top_tfidf_terms_sparse(tfidf_sparse_matrix, labels, label)

for emotion, terms in top_tfidf_terms_sparse.items():
    print(f"Top 5 TF-IDF terms for {emotion}:")
    for term, score in terms:
        print(f"{term}: {score}")
    print()
