import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from keras.models import load_model

def train_sentiment_analysis_model(texts, labels, max_words, embedding_dim, num_epochs, batch_size):

    # Разделение данных на обучающий и тестовый наборы
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    labels_train = labels_train.astype('float32')
    labels_test = labels_test.astype('float32')

    # Токенизация текста
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    sequences_test = tokenizer.texts_to_sequences(texts_test)

    # Дополнение последовательностей для обеспечения одинаковой длины
    max_sequence_length = max(len(sequences_train), len(sequences_test))
    sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length, padding='post').astype('float32')
    sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length, padding='post').astype('float32')

    # Создание модели нейронной сети
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), Precision(), Recall(), F1Score()])

    # Обучение модели
    model.fit(sequences_train, labels_train, epochs=num_epochs, batch_size=batch_size, validation_data=(sequences_test, labels_test))

    # Сохранение модели
    model.save('sentiment_analysis_model_5.h5')

    # Оценка производительности модели
    evaluation = model.evaluate(sequences_test, labels_test)
    print("Test Accuracy:", evaluation[1])
    print("Test Precision:", evaluation[2])
    print("Test Recall:", evaluation[3])
    print("Test F1 Score:", evaluation[4])

    return model
# Загрузка данных из CSV файла
data = pd.read_csv("CyberBullying_Comments_Dataset.csv")
texts = data["Text"]
labels = data["CB_Label"]


# Пример использования функции
max_words = 10000  # Максимальное количество слов в словаре
embedding_dim = 100  # Размерность встраивания слов
num_epochs = 10
batch_size = 64

model = train_sentiment_analysis_model(texts, labels, max_words, embedding_dim, num_epochs, batch_size)


# Загрузка модели
loaded_model = load_model('sentiment_analysis_model.h5')

# Прогнозирование с использованием загруженной модели
predictions = loaded_model.predict(sequences_test)
