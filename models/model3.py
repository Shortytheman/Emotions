import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, TFBertForSequenceClassification, logging as transformers_logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
from tkinter import messagebox

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

# Reduce transformers logging verbosity
transformers_logging.set_verbosity_error()

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the full dataset
print("Loading dataset...")
data = pd.read_csv('../data/emotions_cleaned.csv')
print("Dataset loaded.")

# Use a subset of the data for faster experimentation
subset_size = 200  # Adjust based on your requirements
data = data.sample(n=subset_size, random_state=42)

# Map numerical labels to emotion names
label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['label'] = data['label'].map(label_map)

# Prepare the text and labels
texts = data['text'].values
labels = data['label'].values

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print("Labels encoded.")

# Split the data into training and test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(f"Data split. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = tf.constant(class_weights, dtype=tf.float32)


# Tokenize the text using BERT tokenizer with progress bar
def prepare_bert_data(X_train, X_test, max_length=128):
    print("Tokenizing data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize training data with progress bar
    train_encodings = []
    for text in tqdm(X_train, desc="Tokenizing training data"):
        encodings = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
        train_encodings.append(encodings)

    # Tokenize test data with progress bar
    test_encodings = []
    for text in tqdm(X_test, desc="Tokenizing test data"):
        encodings = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
        test_encodings.append(encodings)

    # Convert lists of encodings to dictionaries
    train_encodings = {key: np.array([enc[key] for enc in train_encodings]) for key in train_encodings[0].keys()}
    test_encodings = {key: np.array([enc[key] for enc in test_encodings]) for key in test_encodings[0].keys()}

    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings, y_test))

    print("Data tokenized.")
    return train_dataset, test_dataset, tokenizer


# Prepare data for BERT model
train_dataset, test_dataset, bert_tokenizer = prepare_bert_data(X_train, X_test)

# Build BERT model
print("Building BERT model...")
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
optimizer = Adam(learning_rate=1e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
metric = SparseCategoricalAccuracy('accuracy')
print("BERT model built.")

# Calculate steps per epoch for the subset
train_samples = len(X_train)
batch_size = 16  # As specified
steps_per_epoch = train_samples // batch_size
print(f"Training samples: {train_samples}")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")

# Custom training loop with early stopping and class weights
epochs = 5

train_dataset = train_dataset.shuffle(10000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=2, restore_best_weights=True)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    # Training loop
    for step, (batch_inputs, batch_labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(batch_inputs, training=True)
            loss = loss_fn(batch_labels, outputs.logits)
            logits = outputs.logits
            # Apply class weights
            weights = tf.gather(class_weights, batch_labels)
            weighted_loss = tf.reduce_mean(loss * weights)
        gradients = tape.gradient(weighted_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 100 == 0:  # Adjust the logging frequency
            print(f"Step {step}: loss = {weighted_loss.numpy()}")

    # Validation loop
    val_loss = 0
    val_steps = 0
    correct_predictions = 0
    total_predictions = 0
    for batch_inputs, batch_labels in test_dataset:
        outputs = model(batch_inputs, training=False)
        loss = loss_fn(batch_labels, outputs.logits)
        val_loss += loss.numpy()
        val_steps += 1
        predictions = tf.argmax(outputs.logits, axis=1, output_type=tf.int64)  # Ensure predictions are int64
        correct_predictions += tf.reduce_sum(
            tf.cast(predictions == tf.cast(batch_labels, tf.int64), tf.float32)).numpy()
        total_predictions += batch_labels.shape[0]

    val_loss /= val_steps
    val_accuracy = correct_predictions / total_predictions
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

    if early_stopping.on_epoch_end(epoch, {'val_accuracy': val_accuracy}):
        print("Early stopping triggered")
        break

# Evaluate model
print("Evaluating model...")
y_pred_bert = []
for batch_inputs, batch_labels in test_dataset:
    outputs = model(batch_inputs, training=False)
    logits = outputs.logits
    predictions = tf.argmax(logits, axis=1).numpy()
    y_pred_bert.extend(predictions)

print("BERT Accuracy:", accuracy_score(y_test, y_pred_bert))
print("BERT Classification Report:\n", classification_report(y_test, y_pred_bert, target_names=label_encoder.classes_))


# Define predict_emotion function
def predict_emotion(message, model, vectorizer, emotion_map):
    cleaned_message = re.sub(r'\W', ' ', message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_tfidf)
    predicted_emotion = emotion_map[prediction[0]]
    return predicted_emotion


# GUI Setup
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

tk.Label(root,
         text=f"Model prediction accuracy: {accuracy_percentage:.2f}%\n\nPlease enter the prediction message, or \"exit\" to exit the loop.",
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
    if not re.match(r'^[A-Za-z\s\'\']+$', text.strip()):
        return False
    return True


with open("../data/newdata.csv", "a") as file:
    for response in user_responses:
        input_text = response['input_text']
        user_emotion = response['user_emotion']

        if is_valid_input(input_text):
            mapped_user_emotion = next(key for key, value in emotion_map.items() if value == user_emotion)
            file.write(f"{input_text}, {mapped_user_emotion}\n")

# Plotting --------------------------------

conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plotting the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_normalized, annot=True, cmap='Blues', xticklabels=emotion_map.values(),
            yticklabels=emotion_map.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report ---------------------------------

report = classification_report(y_test, y_pred, target_names=list(emotion_map.values()), output_dict=True)

# Convert the report to a DataFrame for easier plotting
report_df = pd.DataFrame(report).transpose()

# Filter out support column as it's not needed for this plot
metrics_df = report_df[['precision', 'recall', 'f1-score']].drop('accuracy')

# Plotting the metrics
plt.figure(figsize=(12, 8))
metrics_df.plot(kind='bar', figsize=(12, 8), cmap='viridis')
plt.title('Precision, Recall, and F1-Score for Each Emotion Class')
plt.xlabel('Emotion')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.show()

# Feature Importance ---------------------------------

# Extract feature names
feature_names = vectorizer.get_feature_names_out()

# Extract coefficients
coefficients = model.coef_

# Create a DataFrame to store the coefficients for each class
coef_df = pd.DataFrame(coefficients.T, index=feature_names, columns=emotion_map.values())


# Function to plot top positive and negative features for a given class
def plot_top_features(class_name, top_n=10):
    class_coefficients = coef_df[class_name]
    top_positive_coefficients = class_coefficients.sort_values(ascending=False).head(top_n)
    top_negative_coefficients = class_coefficients.sort_values(ascending=False).tail(top_n)

    top_coefficients = pd.concat([top_positive_coefficients, top_negative_coefficients])

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_coefficients.values, y=top_coefficients.index, palette='viridis')
    plt.title(f'Top {top_n} Positive and Negative Features for {class_name.capitalize()}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()


# Plot top features for each class
for emotion in emotion_map.values():
    plot_top_features(emotion)


# Clean up and save responses ---------------------------------
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