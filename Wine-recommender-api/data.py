import pandas as pd
import numpy as np
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec

def get_data():
    dtype_dict = {str(i):np.float16 for i in range(100)}
    path = "data/wine_dataset_light.csv"
    return pd.read_csv(path, low_memory = True, memory_map = True, dtype=dtype_dict)

def get_params(mode):
    if mode == 'all':
        bad_words = pd.read_csv('data/bad_words.csv').values
        n_grams = Phraser.load("data/ngrams_model.pkl")
        dict_tfidf = np.load('data/my_dict.npy',allow_pickle='TRUE').item()
        word2vec = Word2Vec.load('data/word2vec.model')
        return bad_words, n_grams, dict_tfidf, word2vec
    if mode == 'half':
        bad_words = pd.read_csv('data/bad_words.csv').values
        n_grams = Phraser.load("data/ngrams_model.pkl")
        return bad_words, n_grams

def clean_df(df):
    df = df.drop(columns=['taster_twitter_handle','designation','region_2'])
    df = df.dropna(subset =['variety'])
    df = df.dropna(subset=['price'])
    df = df.dropna(subset=['country'])
    df['region_1']=df['title'].str.extract(r'\((.*?)\)')
    df['region_1'] = df['region_1'].fillna('NR1')
    df['vintage'] = df['vintage'].fillna('NV')
    df['taster_name'] = df['taster_name'].fillna('NTN')
    df = df.reset_index(drop=True)
    return df
