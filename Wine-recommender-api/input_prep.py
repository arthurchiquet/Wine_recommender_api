import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from gensim.models.phrases import Phrases, Phraser
from Wine_recommender_api.utils import remove_punctuation, lemma
from Wine_recommender_api.reviews_prep import get_bad_words, n_grams_words
from gensim.models import Word2Vec
import string

def clean_input(text, bad_words,n_grams):
    text = remove_punctuation(text)
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    text_token = text.split()
    clean_text = [word for word in text_token if not word in bad_words]
    clean_text = lemma(clean_text)
    sno = SnowballStemmer('english')
    clean_text = [sno.stem(word) for word in clean_text]
    clean_text = n_grams[clean_text]
    return clean_text

def transform(text, model, dict_of_tfidf_weightings):
    review = ' '.join(text)
    weighted_review_terms = []
    words = review.split(' ')
    for word in words:
        if word in dict_of_tfidf_weightings.keys():
            tfidf_weighting = dict_of_tfidf_weightings[word]
            word_vector = model.wv.get_vector(word).reshape(1, 100)
            weighted_word_vector = tfidf_weighting * word_vector
            weighted_review_terms.append(weighted_word_vector)
        else:
            continue
    try:
        review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
    except:
        review_vector = []
    return review_vector

