import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Imports-------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import logging
import tkinter as tk
from tkinter import messagebox
from sklearn.utils.class_weight import compute_class_weight

#-------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset preparations-----------------------
logger.info("Loading dataset...")
df = pd.read_csv('data/emotions.csv')
df.drop(columns='Unnamed: 0', inplace=True)

def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

logger.info("Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Mapping labels to readable form for visualization
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
df['label_name'] = df['label'].map(emotion_map)

# Converting labels back to numeric for training
df['label'] = df['label_name'].map({v: k for k, v in emotion_map.items()})

logger.info("Splitting data into train and test sets...")
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
logger.info("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Use bigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logger.info("Text data vectorized successfully.")

# Define the parameter grid for hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5]
}

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=5,  # Use fewer iterations to speed up
    cv=3,  # Use 3-fold cross-validation
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

# Start timer
start_time = time.time()

logger.info("Training Random Forest model...")
random_search.fit(X_train_tfidf, y_train)

# Stop timer
end_time = time.time()
training_time = end_time - start_time
logger.info(f"Model trained successfully in {training_time:.2f} seconds.")

# Get the best parameters and train the best model
best_params = random_search.best_params_
logger.info(f"Best parameters found: {best_params}")
best_model = random_search.best_estimator_
best_model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test_tfidf)
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=list(emotion_map.values())))

def predict_emotion(message, model, vectorizer, emotion_map):
    cleaned_message = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_tfidf)
    predicted_emotion = emotion_map[prediction[0]]
    return predicted_emotion

# GUI SETUP-----------------------------------------

flag = True

def submit_input(event=None):
    global flag
    prediction_text = prediction_entry.get()
    if prediction_text.lower() == "exit":
        flag = False
        root.quit()
        return
    
    predicted_emotion = predict_emotion(prediction_text, best_model, vectorizer, emotion_map)
    output_text = f"Predicted Emotion: {predicted_emotion}\n"
    output_label.config(text=output_text)
    # Clear the entry field
    prediction_entry.delete(0, tk.END)

def reset_gui():
    output_label.config(text="")
    prediction_entry.delete(0, tk.END)
    prediction_entry.focus()

total_predictions = len(y_test)
correct_predictions = (y_test == y_pred).sum()
accuracy_percentage = (correct_predictions / total_predictions) * 100

while flag:
    root = tk.Tk()
    root.title("Prediction GUI")

    # Set font size for labels and buttons
    label_font = ("Arial", 24)
    button_font = ("Arial", 20)

    tk.Label(root, text=f"Model prediction accuracy: {accuracy_percentage:.2f}%\n\nPlease enter the prediction message, or \"exit\" to exit the loop.",
             font=label_font).pack(pady=5)
    prediction_entry = tk.Entry(root, font=label_font)
    prediction_entry.pack(pady=5)

    submit_button = tk.Button(root, text="Submit", command=submit_input, font=button_font)
    submit_button.pack(pady=10)

    reset_button = tk.Button(root, text="Reset", command=reset_gui, font=button_font)
    reset_button.pack(pady=10)

    output_label = tk.Label(root, text="", fg="blue", font=label_font)
    output_label.pack(pady=10)

    # Bind the <Return> key to submit_input function
    root.bind("<Return>", submit_input)

    root.mainloop()

    if not flag:
        break  # Exit the loop if the user exits

    # Display accuracy for the last prediction
    if y_test is not None:
        last_prediction_text = prediction_text
        last_predicted_emotion = predicted_emotion
        last_actual_emotion = emotion_map[y_test[-1]]
        last_prediction_accuracy = "Correct" if last_predicted_emotion == last_actual_emotion else "Incorrect"
        logger.info(f"Input: {last_prediction_text}, Predicted Emotion: {last_predicted_emotion}, Actual Emotion: {last_actual_emotion}, Accuracy: {last_prediction_accuracy}")

### Fitting 3 folds for each of 5 candidates, totalling 15 fits ###
# Best parameters found: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': None}
# Accuracy: 0.8814687747414889
# Model trained successfully in 377.94 seconds.