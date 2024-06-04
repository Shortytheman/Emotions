import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
import tkinter as tk
from tkinter import messagebox
from sklearn.utils.class_weight import compute_class_weight
import time
from sklearn import tree


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Loading dataset...")
df = pd.read_csv('../data/emotions_cleaned.csv')


emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
df['label_name'] = df['label'].map(emotion_map)


df['label'] = df['label_name'].map({v: k for k, v in emotion_map.items()})


df['cleaned_text'] = df['cleaned_text'].fillna('')

logger.info("Splitting data into train and test sets...")
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logger.info("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logger.info("Text data vectorized successfully.")


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)


logger.info("Training Decision Tree model...")
start_time = time.time()
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)


model_for_plot = DecisionTreeClassifier(max_depth=5, random_state=42)
model_for_plot.fit(X_train_tfidf, y_train)

logger.info("Model trained successfully.")
logger.info(f"Training time: {time.time() - start_time} seconds")


logger.info("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)

logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=list(emotion_map.values())))


def predict_emotion(message, model, vectorizer, emotion_map):
    cleaned_message = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_tfidf)
    predicted_emotion = emotion_map[prediction[0]]
    return predicted_emotion


flag = True
user_responses = []

def submit_input(event=None):
    global flag, prediction_text
    prediction_text = prediction_entry.get("1.0", tk.END).strip()
    if prediction_text.lower() == "exit":
        flag = False
        root.quit()
        return

    predicted_emotion = predict_emotion(prediction_text, model, vectorizer, emotion_map)
    output_text = f"Predicted Emotion: {predicted_emotion}\n"
    output_label.config(text=output_text)
    
    submit_button.config(state=tk.DISABLED)
    prediction_entry.config(state=tk.DISABLED)
    
    emotion_label.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")
    for button in emotion_buttons:
        button.grid(pady=2, padx=2)

def reset_gui():
    output_label.config(text="")
    prediction_entry.config(state=tk.NORMAL)
    prediction_entry.delete("1.0", tk.END)
    prediction_entry.focus()
    submit_button.config(state=tk.NORMAL)
    emotion_label.grid_remove()
    for button in emotion_buttons:
        button.grid_remove()

def save_response(emotion):
    global prediction_text
    user_response = {
        "input_text": prediction_text,
        "predicted_emotion": predict_emotion(prediction_text, model, vectorizer, emotion_map),
        "user_emotion": emotion
    }
    user_responses.append(user_response)
    print(f"Response saved: {user_response}")
    messagebox.showinfo("Saved", f"Your response has been saved: {emotion}")
    reset_gui()

def exit_application():
    global flag
    flag = False
    root.quit()

total_predictions = len(y_test)
correct_predictions = (y_test == y_pred).sum()
accuracy_percentage = (correct_predictions / total_predictions) * 100

root = tk.Tk()
root.title("Prediction GUI")

label_font = ("Arial", 24)
button_font = ("Arial", 20)

tk.Label(root, text=f"Model prediction accuracy: {accuracy_percentage:.2f}%\n\nPlease enter the prediction message, or \"exit\" to exit the loop.",
         font=label_font).grid(row=0, column=0, columnspan=3, pady=5)
prediction_entry = tk.Text(root, font=label_font, width=50, height=5)
prediction_entry.grid(row=1, column=0, columnspan=3, pady=5)

submit_button = tk.Button(root, text="Submit", command=submit_input, font=button_font)
submit_button.grid(row=2, column=0, padx=2, pady=5, sticky="ew")

reset_button = tk.Button(root, text="Reset", command=reset_gui, font=button_font)
reset_button.grid(row=2, column=1, padx=2, pady=5, sticky="ew")

exit_button = tk.Button(root, text="Exit", command=exit_application, font=button_font)
exit_button.grid(row=2, column=2, padx=2, pady=5, sticky="ew")

output_label = tk.Label(root, text="", fg="blue", font=label_font)
output_label.grid(row=3, column=0, columnspan=3, pady=10)

emotion_label = tk.Label(root, text="Choose the correct emotion based on your input:", font=label_font)

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotion_buttons = []
for i, emotion in enumerate(emotions):
    button = tk.Button(root, text=emotion, command=lambda e=emotion: save_response(e), font=button_font)
    emotion_buttons.append(button)
    button.grid(row=5 + i // 3, column=i % 3, pady=2, padx=2)
    button.grid_remove()

root.mainloop()

total_predictions = len(user_responses)
correct_predictions = sum(1 for response in user_responses if response['predicted_emotion'] == response['user_emotion'])
accuracy_percentage = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

accuracy_message = f"The bot has been {accuracy_percentage:.2f}% accurate based on your responses."
messagebox.showinfo("Accuracy", accuracy_message)

def is_valid_input(text):
    words = text.split()
    if len(words) < 5 or len(words) > 35:
        return False
    if not re.match(r'^[A-Za-z\s,!?\'\']+$', text.strip()):
        return False
    return True

with open("../data/newdata.csv", "a") as file:
    for response in user_responses:
        input_text = response['input_text']
        user_emotion = response['user_emotion']
        
        if is_valid_input(input_text):
            mapped_user_emotion = next(key for key, value in emotion_map.items() if value == user_emotion)
            file.write(f"{input_text}, {mapped_user_emotion}\n")



conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_normalized, annot=True, cmap='Blues', xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()




y_test_names = y_test.map(emotion_map)
y_pred_names = pd.Series(y_pred).map(emotion_map)

report = classification_report(y_test, y_pred, target_names=list(emotion_map.values()), output_dict=True)


report_df = pd.DataFrame(report).transpose()


metrics_df = report_df[['precision', 'recall', 'f1-score']].drop('accuracy')


plt.figure(figsize=(12, 8))
metrics_df.plot(kind='bar', figsize=(12, 8), cmap='viridis')
plt.title('Precision, Recall, and F1-Score for Each Emotion Class')
plt.xlabel('Emotion')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.show()



plt.figure(figsize=(10, 10)) 
tree.plot_tree(model_for_plot, filled=True, rounded=True, class_names=list(emotion_map.values()), feature_names=vectorizer.get_feature_names_out())
plt.title("Decision Tree Visualization")
plt.show()

report = classification_report(y_test, y_pred, target_names=list(emotion_map.values()), output_dict=True)


report_df = pd.DataFrame(report).transpose()


metrics_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore').drop(columns=['support'])


mean_precision = metrics_df['precision'].mean()
mean_recall = metrics_df['recall'].mean()
mean_f1_score = metrics_df['f1-score'].mean()


print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1-Score: {mean_f1_score:.4f}")
