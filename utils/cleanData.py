import pandas as pd
import re

# Define the cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()  # Lowercase text
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words that are 1-2 letters long
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip whitespace
    return text

# Read the data from the original CSV file
data = pd.read_csv('../data/emotions.csv')

# Drop the unnecessary column
data.drop(columns='Unnamed: 0', inplace=True)

# Clean the text data
data['cleaned_text'] = data['text'].apply(clean_text)

# Handle any remaining NaNs in 'cleaned_text'
data['cleaned_text'] = data['cleaned_text'].fillna('')

# Retain only the 'cleaned_text' and 'label' columns, renaming 'cleaned_text' back to 'text'
cleaned_data = data[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'})

# Write the cleaned data to a new CSV file
cleaned_data.to_csv('../data/emotions_cleaned.csv', index=False)

print("Data cleaned and saved to '../data/emotions_cleaned.csv'")
