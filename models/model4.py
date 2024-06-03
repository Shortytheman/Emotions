import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy.sparse import csr_matrix
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and prepare dataset
logger.info("Loading dataset...")
df = pd.read_csv('../data/emotions_cleaned.csv')

# Map labels to readable form for visualization
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
df['label_name'] = df['label'].map(emotion_map)

# Converting labels back to numeric for training
df['label'] = df['label_name'].map({v: k for k, v in emotion_map.items()})

# Check for NaN values and handle them
df['cleaned_text'] = df['cleaned_text'].fillna('')

logger.info("Splitting data into train and test sets...")
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
logger.info("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))  # Reduce max_features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logger.info("Text data vectorized successfully.")

# Convert labels to one-hot encoding
y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Dense(512, input_shape=(2000,), activation='relu'))  # Adjust input_shape to match max_features
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotion_map), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
logger.info("Training the model...")
history = model.fit(X_train_tfidf, y_train_enc,
                    epochs=1,
                    batch_size=32,
                    validation_data=(X_test_tfidf, y_test_enc),
                    verbose=1)
logger.info("Model trained successfully.")

# Evaluate the model
logger.info("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_tfidf, y_test_enc, verbose=1)
logger.info(f"Accuracy: {accuracy}")

# Get classification report
y_pred_enc = model.predict(X_test_tfidf)
y_pred = np.argmax(y_pred_enc, axis=1)

from sklearn.metrics import classification_report
logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=list(emotion_map.values())))
