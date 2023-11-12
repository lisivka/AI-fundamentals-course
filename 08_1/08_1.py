# python -m spacy download en_core_web_sm

# ===========================================================================
#import spacy library
import spacy

# # @test_get_segment
# def get_segment(some_text):
#   nlp = spacy.load("en_core_web_sm")
#   #take string
#   doc = nlp(some_text)
#   sentences = []
#   #to print sentences
#   for sentence in doc.sents:
#     sentences.append(sentence)
#   return sentences
# text = '''Natural Language Processing (NLP) is a part of AI (artificial intelligence) that deals with understanding and processing of human language.
# In real time, majority of data exists in the unstructured form such us text, videos, images.
# Mass of data in unstructured category, will be in textual form.
# To process this textual data's with machine learning algorithms, NLP comes in to play.
# NLP use cases are Language translation, Speech recognition, Hiring and Recruitment, Chat Bot, Sentimental analysis and so on.'''
#
# segment=get_segment(text)
# print(*segment, sep="\n")
#
# segment=get_segment("Natural. Language. Processing")
# print(len(list(segment)))


# # ===========================================================================
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
# nlp = English ()
# tokenizer = Tokenizer(nlp.vocab)
#
# text = """Natural Language Processing (NLP) is a part of AI (artificial intelligence)
#           that deals with understanding and processing of human language."""
# text = "Natural Language Processing IS THE best choice for the learning"
#
# # @test_get_tokens
# def get_tokens(some_text):
#   nlp = English()
#   tokenizer = Tokenizer(nlp.vocab)
#   tokens_nlp = tokenizer(some_text)
#   # tokens = []
#   # for token in tokens_nlp:
#   #   tokens.append(token.text)
#   return tokens_nlp
#
# tokens = get_tokens(text)
# print(*tokens, sep=" | ")
#
# # ===========================================================================
# def remove_stop_words(tokens):
#   filtered_sentence =[]
#
#   for word in tokens:
#       if word.is_stop == False:
#           filtered_sentence.append(word)
#
#   return filtered_sentence
#
#
# filtered_sentence = remove_stop_words(tokens)
# print(*filtered_sentence, sep=" | ")
#
#
#
#
# # import module for stop words removing
#
# from spacy.lang.en.stop_words import STOP_WORDS
# def remove_stop_words(tokens):
#     filtered_sentence = []
#
#     for word in tokens:
#         if word.text.lower() not in STOP_WORDS:
#             filtered_sentence.append(word.text)
#
#     return filtered_sentence
#
# filtered_sentence = remove_stop_words(tokens)
# print(*filtered_sentence, sep=" | ")


# ===========================================================================
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

def stemm_lemmatization(text):
    # Токенизация слов
    words = word_tokenize(text)

    # Инициализация стеммера и лемматизатора
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    # Список для хранения лемм
    lemmatized_words = []

    for word in words:
        # Стемминг
        stemmed_word = porter_stemmer.stem(word)
        # Лемматизация
        lemmatized_word = wordnet_lemmatizer.lemmatize(word, pos='v')  # pos='v' для указания, что это глагол

        lemmatized_words.append(lemmatized_word)

    return lemmatized_words

# Пример использования
text = "Stemming and lemmatization are important natural language processing techniques."
result = stemm_lemmatization(text)
print(result)
