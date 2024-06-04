import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

data = pd.read_csv('../data/emotions.csv')

data.drop(columns='Unnamed: 0', inplace=True)

data['cleaned_text'] = data['text'].apply(clean_text)

data.to_csv('../data/emotions_cleaned.csv', index=False)

print("Data cleaned and saved to '../data/emotions_cleaned.csv'")
