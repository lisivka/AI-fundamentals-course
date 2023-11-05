import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Load and preprocess your dataset (CyberBullying_Comments_Dataset.csv)
# You can use a library like pandas to load your dataset.
# Preprocess your data, split it into train and test sets, and tokenize it.

# Example data: text reviews and sentiment labels
texts = ["This is a positive review.", "This is a negative review."]
labels = [1, 0]  # 1 for positive, 0 for negative

# Hyperparameters
max_words = 10000  # Maximum number of words to tokenize
embedding_dim = 100  # Dimension of word embeddings
num_epochs = 10
batch_size = 32

# Tokenize the text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to ensure uniform length
X = pad_sequences(sequences)

# Build the neural network model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))  # Binary sentiment classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the sentiment analysis model
model.fit(X, np.array(labels), epochs=num_epochs, batch_size=batch_size)

# Test the model

# Example new text reviews
new_texts = ["I love this product.", "This is terrible."]

# Tokenize and pad the new text reviews
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=X.shape[1])

# Use the trained model to predict sentiments
predictions = model.predict(new_X)

# Convert the predictions to binary labels (0 or 1)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

# Print the predicted sentiments for new text reviews
for i, text in enumerate(new_texts):
    sentiment = "positive" if binary_predictions[i] == 1 else "negative"
    print(f"Text: {text} | Predicted Sentiment: {sentiment}")
