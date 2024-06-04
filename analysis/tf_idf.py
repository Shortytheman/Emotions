import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict


data = pd.read_csv("../data/emotions_cleaned.csv")


data["cleaned_text"] = data["cleaned_text"].fillna("")


tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")


tfidf_sparse_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])


label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
data["label"] = data["label"].map(label_map)

labels = data["label"].values

excluded_words = {"really", "little"}


def get_top_tfidf_terms_sparse(matrix, labels, label, n=20):
    label_indices = labels == label
    label_matrix = matrix[label_indices]
    mean_tfidf = label_matrix.mean(axis=0)
    mean_tfidf = np.array(mean_tfidf).flatten()
    top_terms = [
        (tfidf_vectorizer.get_feature_names_out()[i], mean_tfidf[i])
        for i in mean_tfidf.argsort()[-n:][::-1]
    ]
    return top_terms


top_tfidf_terms_sparse = {}
for label in np.unique(labels):
    top_tfidf_terms_sparse[label] = get_top_tfidf_terms_sparse(
        tfidf_sparse_matrix, labels, label
    )


term_scores_across_emotions = defaultdict(float)
term_count_across_emotions = defaultdict(int)

for emotion, terms in top_tfidf_terms_sparse.items():
    for term, score in terms:
        term_scores_across_emotions[term] += score
        term_count_across_emotions[term] += 1


threshold = 0.06


common_high_value_terms = {
    term
    for term, count in term_count_across_emotions.items()
    if count == len(label_map) and term_scores_across_emotions[term] > threshold
}


def get_top_unique_tfidf_terms_sparse(matrix, labels, label, n=5):
    label_indices = labels == label
    label_matrix = matrix[label_indices]
    mean_tfidf = label_matrix.mean(axis=0)
    mean_tfidf = np.array(mean_tfidf).flatten()

    top_terms = []
    for idx in mean_tfidf.argsort()[-n * 3 :][::-1]:
        term = tfidf_vectorizer.get_feature_names_out()[idx]
        if term not in common_high_value_terms and term not in excluded_words:
            top_terms.append((term, mean_tfidf[idx]))
        if len(top_terms) == n:
            break
    return top_terms


top_unique_tfidf_terms_sparse = {}
for label in np.unique(labels):
    top_unique_tfidf_terms_sparse[label] = get_top_unique_tfidf_terms_sparse(
        tfidf_sparse_matrix, labels, label
    )


for emotion, terms in top_unique_tfidf_terms_sparse.items():
    print(f"Top 5 unique TF-IDF terms for {emotion}:")
    for term, score in terms:
        print(f"{term}: {score}")
    print()
