import numpy as np
import pandas as pd
import string as st
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

def compute_mean_embedding(model=None,list=None):
    clean_list=[model.wv[word] for word in list if word in model.wv.vocab]
    return np.mean(clean_list,axis=0)

def compute_distance(reviews_vector, input_vector, distance='euclidean'):
    ### compute euclidian distance between input wine and other wines
    if distance =='euclidean':
        distance_calc = euclidean_distances(reviews_vector, input_vector)
    elif distance == 'cosine':
        distance_calc = cosine_distances(reviews_vector, input_vector)
    distance_float = np.array([l[0] for l in distance_calc])
    return distance_float

def remove_punctuation(text):
    for punctuation in st.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def lemma(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    return lemmatized

def lowercase(word):
    return word.lower()

def split_url(title):
    title = title.lower()
    for i in ['(',')']:
        title = title.replace(i,'')
    title_list = title.split()
    url = ''
    for i in title_list:
        url += i + "+"
    return url[:-1]

def clean_reviews(text, bad_words, n_grams):
    text = remove_punctuation(text)
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    text_token = text.split()
    clean_text = [word for word in text_token if not word in bad_words]
    sno = SnowballStemmer('english')
    clean_text = lemma(clean_text)
    clean_text = [sno.stem(word) for word in clean_text]
    clean_text = n_grams[clean_text]
    return clean_text
