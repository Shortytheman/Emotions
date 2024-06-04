import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv("../data/emotions_cleaned.csv")

# Map numerical labels to emotion names
label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
data["emotion"] = data["label"].map(label_map)

# Count the frequency of each emotion
emotion_counts = data["emotion"].value_counts().sort_index()

# Ensure all labels are present in the frequency count
all_labels = pd.Index(["sadness", "joy", "love", "anger", "fear", "surprise"])
emotion_counts = emotion_counts.reindex(all_labels, fill_value=0)

# Plot the frequency distribution
plt.figure(figsize=(10, 6))
emotion_counts.plot(kind="bar", color="skyblue")
plt.title("Frequency Distribution of Emotions")
plt.xlabel("Emotion")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()
