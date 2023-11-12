# python -m spacy download en_core_web_sm

# ===========================================================================
#import spacy library
import spacy

# @test_get_segment
def get_segment(some_text):
  nlp = spacy.load("en_core_web_sm")
  #take string
  doc = nlp(some_text)
  sentences = []
  #to print sentences
  for sentence in doc.sents:
    sentences.append(sentence)
  return sentences
text = '''Natural Language Processing (NLP) is a part of AI (artificial intelligence) that deals with understanding and processing of human language.
In real time, majority of data exists in the unstructured form such us text, videos, images.
Mass of data in unstructured category, will be in textual form.
To process this textual data's with machine learning algorithms, NLP comes in to play.
NLP use cases are Language translation, Speech recognition, Hiring and Recruitment, Chat Bot, Sentimental analysis and so on.'''

segment=get_segment(text)
print(*segment, sep="\n")

segment=get_segment("Natural. Language. Processing")
print(len(list(segment)))


# ===========================================================================
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English ()
tokenizer = Tokenizer(nlp.vocab)

text = """Natural Language Processing (NLP) is a part of AI (artificial intelligence)
          that deals with understanding and processing of human language."""
text = "Natural Language Processing IS THE best choice for the learning"

# @test_get_tokens
def get_tokens(some_text):
  nlp = English()
  tokenizer = Tokenizer(nlp.vocab)
  tokens_nlp = tokenizer(some_text)
  # tokens = []
  # for token in tokens_nlp:
  #   tokens.append(token.text)
  return tokens_nlp

tokens = get_tokens(text)
print(*tokens, sep=" | ")

# ===========================================================================
def remove_stop_words(tokens):
  filtered_sentence =[]

  for word in tokens:
      if word.is_stop == False:
          filtered_sentence.append(word)

  return filtered_sentence


filtered_sentence = remove_stop_words(tokens)
print(*filtered_sentence, sep=" | ")




# import module for stop words removing

from spacy.lang.en.stop_words import STOP_WORDS
def remove_stop_words(tokens):
    filtered_sentence = []

    for word in tokens:
        if word.text.lower() not in STOP_WORDS:
            filtered_sentence.append(word.text)

    return filtered_sentence

filtered_sentence = remove_stop_words(tokens)
print(*filtered_sentence, sep=" | ")


# ===========================================================================
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

def stemm_lemmatization(text):
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(text)
  stemm_lemm = []
  for token in doc:
    stemm_lemm.append(token.lemma_)

  return  stemm_lemm

doc = "Natural Language Processing IS THE best choice for the learning"
print(stemm_lemmatization(doc))



def stemm_lemmatization2(text):
    tokens = word_tokenize(text)
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    stemmed_words = []
    for token in tokens:
        stemmed_word = porter_stemmer.stem(str(token).lower())
        lemmatized_word = wordnet_lemmatizer.lemmatize(str(token).lower(), pos="v")  # pos='v' для дієслів
        stemmed_words.append(stemmed_word)
        lemmatized_words.append(lemmatized_word)
    return stemmed_words, lemmatized_words


text = "Natural Language Processing IS THE best choice for the learning"
stemmed_words, lemmatized_words = stemm_lemmatization2(text)
print("stemmed_words   ", stemmed_words)
print("lemmatized_words", lemmatized_words)

# ===========================================================================

import spacy


# @test_part_of/
def part_of(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    word_tags = {}
    for word in doc:
        word_tags[word.text] = word.pos_

    return word_tags

text = "Natural Language Processing IS THE best choice for the learning"
word_tags= part_of(text)
print(word_tags)