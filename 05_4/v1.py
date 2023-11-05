import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model


def train_sentiment_analysis_model(texts, labels, max_words, embedding_dim, num_epochs, batch_size):
    """
    Train a neural network model for sentiment analysis using word embeddings.

    Args:
        texts (list): List of text reviews.
        labels (list): List of corresponding sentiment labels (0 or 1).
        max_words (int): Maximum number of words to tokenize.
        embedding_dim (int): Dimension of word embeddings.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    # Tokenize and pad the input data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_words)

    # Convert labels to NumPy array
    labels = np.array(labels)

    # Create the model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(data, labels, epochs=num_epochs, batch_size=batch_size, verbose=0)

    # Save the trained model
    model.save('sentiment_analysis_model.h5')


# Load the data
data = pd.read_csv('CyberBullying_Comments_Dataset.csv')[::50]
print(data.head())

# Split the data into texts and labels
texts = data['Text'].tolist()
labels = data['CB_Label'].tolist()

# Split the data into training and validation sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(f"Training set: {len(train_texts)}")
print(f"Validation set: {len(test_texts)}")

# Hyperparameters
max_words = 10000
embedding_dim = 1000
num_epochs = 100
batch_size = 32

# # Train the sentiment analysis model
train_sentiment_analysis_model(train_texts, train_labels, max_words, embedding_dim, num_epochs, batch_size)

# Test the model
# Load your model file
model = load_model('sentiment_analysis_model.h5')

# Tokenize and pad the new text reviews
tokenizer = Tokenizer(num_words=max_words)

new_sequences = tokenizer.texts_to_sequences(test_texts)
new_data = pad_sequences(new_sequences, maxlen=max_words)
print(f"Shape of new data: {new_data.shape}")

# Use the trained model to predict sentiments
predictions = model.predict(new_data)


# Convert the predictions to binary labels (0 or 1)
binary_labels = (predictions > 0.5).astype(int)



# Print the predicted sentiments for new text reviews
for i, text  in enumerate(test_texts):
    print(f" i = {i} ")
    sentiment = "Positive" if binary_labels[i] == 0 else "Negative"
    expected = "Positive" if test_labels[i] == 0 else "Negative"
    print(f"Review: {text}\nSentiment: {sentiment}\nExpected: {test_labels[i]} {expected}\n")

from sklearn.metrics import confusion_matrix, accuracy_score
# Calculate the confusion matrix
confusion = confusion_matrix(test_labels, binary_labels)


# Extract TP, TN, FP, FN
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]

# Calculate accuracy
accuracy = accuracy_score(test_labels, binary_labels)


# Print the results
print("Confusion Matrix:")
print(confusion)
print(f"True Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Accuracy: {accuracy}")




