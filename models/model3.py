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
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

# Reduce transformers logging verbosity
transformers_logging.set_verbosity_error()

# Load the cleaned dataset
print("Loading dataset...")
data = pd.read_csv('../data/emotions_cleaned.csv')
print("Dataset loaded.")

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

# Use a larger subset for more meaningful training
X_train = X_train[:1000]
y_train = y_train[:1000]
X_test = X_test[:200]
y_test = y_test[:200]


# Tokenize the text using BERT tokenizer
def prepare_bert_data(X_train, X_test, max_length=128):
    print("Tokenizing data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=max_length)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
    print("Data tokenized.")
    return train_dataset, test_dataset, tokenizer


# Prepare data for BERT model
train_dataset, test_dataset, bert_tokenizer = prepare_bert_data(X_train, X_test)

# Build BERT model
print("Building BERT model...")
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
optimizer = Adam(learning_rate=3e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
metric = SparseCategoricalAccuracy('accuracy')
print("BERT model built.")

# Custom training loop
epochs = 3
batch_size = 16

train_dataset = train_dataset.shuffle(1000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    # Training loop
    for step, (batch_inputs, batch_labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(batch_inputs, training=True)
            loss = loss_fn(batch_labels, outputs.logits)
            logits = outputs.logits
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.numpy()}")

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
