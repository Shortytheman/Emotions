import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load NLTK English stopwords
stop_words = set(stopwords.words('english'))

data = pd.read_csv('../data/emotions.csv')

label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['emotion'] = data['label'].map(label_map)

data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
plt.hist(data['word_count'], bins=range(0, 180, 5), color='skyblue', edgecolor='black')
plt.title('Distribution of Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xticks(range(0, 180, 10))
plt.grid(True)
plt.show()

word_count_stats = data['word_count'].describe()
print("Basic Statistics of Word Counts:")
print(word_count_stats)

median_word_counts = data.groupby('emotion')['word_count'].median()
average_word_counts = data.groupby('emotion')['word_count'].mean()

word_count_stats_by_emotion = pd.DataFrame({
    'Median Word Count': median_word_counts,
    'Average Word Count': average_word_counts
})

print("\nMedian and Average Word Counts by Emotion:")
print(word_count_stats_by_emotion)

def get_top_n_words_by_emotion(data, label, n=20):
    text_data = data[data['label'] == label]['text']
    all_words = ' '.join(text_data).split()
    word_counts = Counter([word for word in all_words if word.lower() not in stop_words])
    top_n_words = word_counts.most_common(n)
    return top_n_words

top_words_by_emotion = {}
for label, emotion in label_map.items():
    top_words_by_emotion[emotion] = get_top_n_words_by_emotion(data, label)

for emotion, words in top_words_by_emotion.items():
    words_df = pd.DataFrame(words, columns=['Word', 'Frequency'])
    plt.figure(figsize=(12, 6))
    plt.bar(words_df['Word'], words_df['Frequency'], color='skyblue')
    plt.title(f'Top 20 Most Used Words for Emotion: {emotion} (Excluding Stopwords)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
