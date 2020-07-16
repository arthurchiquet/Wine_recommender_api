import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from gensim.models.phrases import Phrases, Phraser
from Wine_recommender_api.utils import remove_punctuation, lemma
from Wine_recommender_api.data import get_data, clean_df
import string

stop_words_manual = ['wine','bottle','drink','cabernet','sauvignon','franc','champagne']

def without_punctuation(reviews):
    return [remove_punctuation(review) for review in reviews]

def lowercase(reviews):
    return [review.lower() for review in reviews]

def remove_numbers(reviews):
    return [''.join(word for word in review if not word.isdigit()) for review in reviews]

def get_bad_words(df):
    bad_words_1 = df['province'].str.lower().unique().tolist()
    bad_words_2 = df['variety'].str.lower().unique().tolist()
    return bad_words_1 + bad_words_2

def remove_bad_words(reviews,bad_words):
    stop_words = set(stopwords.words('english'))
    warning_words = list(stop_words) + bad_words
    reviews_token = [word_tokenize(review) for review in reviews]
    without_stopwords = [[word for word in review if not word in warning_words] for review in reviews_token]
    return without_stopwords

def lemmatiz(reviews):
    return [lemma(review) for review in reviews]

def snowball(reviews):
    sno = SnowballStemmer('english')
    stemmed_data=[[sno.stem(word) for word in review] for review in reviews]
    return stemmed_data

def n_grams_words(reviews):
    phrases = Phrases(reviews)
    phrases = Phrases(phrases[reviews])
    ngrams = Phraser(phrases)
    phrased_sentences = [ngrams[review] for review in reviews]
    return np.array(phrased_sentences)

def desc_prepoc(reviews, bad_words):
    reviews = without_punctuation(reviews)
    reviews = lowercase(reviews)
    reviews = remove_numbers(reviews)
    reviews = remove_bad_words(reviews,bad_words)
    reviews = lemmatiz(reviews)
    reviews = snowball(reviews)
    reviews = n_grams_words(reviews)
    return reviews
