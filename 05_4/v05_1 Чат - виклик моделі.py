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
from .v5_1_ import prepare_data, create_model, train_model




# Загрузка данных из CSV файла
data = pd.read_csv("CyberBullying_Comments_Dataset.csv")
texts = data["Text"]
labels = data["CB_Label"]
name_model = "CyberBullying_model_5.h5"

# Загрузка модели
# loaded_model = load_model('sentiment_analysis_model.h5')
name_model = "CyberBullying_model_5.h5"
loaded_model = load_model(name_model)

# Подготовка данных
max_words = 10000  # Максимальное количество слов в словаре
sequences_train, labels_train, sequences_test, labels_test = prepare_data(texts, labels, max_words)
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Прогнозирование с использованием загруженной модели
predictions = loaded_model.predict(sequences_test)


# Convert the predictions to binary labels (0 or 1)
binary_labels = (predictions > 0.5).astype(int)



# Print the predicted sentiments for new text reviews
for i, text in enumerate(texts_test):
    sentiment = "Positive" if binary_labels[i] == 0 else "Negative"
    expected = "Positive" if labels_test.iloc[i] == 0 else "Negative"
    print(f"Review: {text}\nSentiment: {sentiment}\nExpected: {labels_test.iloc[i]} {expected}\n")
