import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model


def train_sentiment_analysis_model(texts, labels, max_words, embedding_dim,
                                   num_epochs, batch_size):
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

    # Create the model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(data, labels, epochs=num_epochs, batch_size=batch_size)

    # Save the trained model
    model.save('sentiment_analysis_model.h5')


# Example data: text reviews and sentiment labels
texts = ["I loved this movie!", "It was a terrible experience.",
         "The acting was great.", "Not worth my time."]
labels = [1, 0, 1, 0]

# Hyperparameters
max_words = 10000
embedding_dim = 100
num_epochs = 10
batch_size = 32

# Train the sentiment analysis model
train_sentiment_analysis_model(texts, labels, max_words, embedding_dim,
                               num_epochs, batch_size)

# Test the model
# Load your model file
model = load_model('sentiment_analysis_model.h5')

# Example new text reviews
new_texts = ["This is an amazing product.", "I regret buying this.",
             "The service was fantastic."]

# Tokenize and pad the new text reviews
tokenizer = Tokenizer(num_words=max_words)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data = pad_sequences(new_sequences, maxlen=max_words)

# Use the trained model to predict sentiments
predictions = model.predict(new_data)

# Convert the predictions to binary labels (0 or 1)
binary_labels = (predictions > 0.5).astype(int)

# Print the predicted sentiments for new text reviews
for i, text in enumerate(new_texts):
    sentiment = "Positive" if binary_labels[i] == 1 else "Negative"
    print(f"Review: {text}\nSentiment: {sentiment}\n")
