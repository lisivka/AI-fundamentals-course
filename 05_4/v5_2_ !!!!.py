import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from keras.models import load_model

def prepare_data(texts, labels, max_words):
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    labels_train = labels_train.astype('float32')
    labels_test = labels_test.astype('float32')

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    sequences_test = tokenizer.texts_to_sequences(texts_test)

    max_sequence_length = max(len(sequences_train), len(sequences_test))
    sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length, padding='post').astype('float32')
    sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length, padding='post').astype('float32')

    return sequences_train, labels_train, sequences_test, labels_test

def create_model(max_words, embedding_dim, max_sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))  # Удалено Flatten и один слой Dropout
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Precision(), Recall(), F1Score()])

    return model
def train_model(model, sequences_train, labels_train,  file_model, num_epochs=1, batch_size=64):
    model.fit(sequences_train, labels_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    # Save the trained model
    model.save(file_model)

def evaluate_model(model, sequences_test, labels_test):
    # Evaluation of the model
    evaluation = model.evaluate(sequences_test, labels_test)
    print("Test Accuracy:", evaluation[1])
    print("Test Precision:", evaluation[2])
    print("Test Recall:", evaluation[3])
    print("Test F1 Score:", evaluation[4])


def run_train_sentiment_analysis_model(texts, labels, file_model, max_words, embedding_dim, num_epochs, batch_size):

    sequences_train, labels_train, sequences_test, labels_test = prepare_data(texts, labels, max_words)
    model = create_model(max_words, embedding_dim, len(sequences_train[0]))
    train_model(model, sequences_train, labels_train, file_model, num_epochs, batch_size)
    evaluate_model(model, sequences_test, labels_test)



import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    # stop_words = set(stopwords.words('english'))
    # words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Загрузка данных из CSV файла
data = pd.read_csv("CyberBullying_Comments_Dataset.csv")
texts = data["Text"]
labels = data["CB_Label"]

# Нормализация текстовых данных
texts = texts.apply(preprocess_text)



# Hyperparameters
max_words = 10000  # Максимальна кількість слів у словнику
embedding_dim = 100  # Розмірність вбудовування слів
num_epochs = 1  # Кількість епох навчання
batch_size = 32  # Розмір пакета


file_model = "CyberBullying_model_5.keras"
model = run_train_sentiment_analysis_model(texts, labels, file_model, max_words, embedding_dim, num_epochs, batch_size)



# Load model file
loaded_model = load_model(file_model)

sequences_train, labels_train, sequences_test, labels_test = prepare_data(texts, labels, max_words)
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
#

# Predictions using the loaded model
predictions = loaded_model.predict(sequences_test)


# Convert the predictions to binary labels (0 or 1)
binary_labels = (predictions > 0.5).astype(int)



# Print the predicted sentiments for new text reviews
for i, text in enumerate(texts_test):
    sentiment = "Positive" if binary_labels[i] == 0 else "Negative"
    expected = "Positive" if labels_test.iloc[i] == 0 else "Negative"
    print(f"Review: {text}\nSentiment: {sentiment}\nExpected: {labels_test.iloc[i]} {expected}\n")
