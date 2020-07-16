import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from Wine_recommender_api.data import get_data, clean_df
from Wine_recommender_api.reviews_prep import desc_prepoc
from sklearn.feature_extraction.text import TfidfVectorizer
from Wine_recommender_api.utils import compute_mean_embedding, compute_distance
from sklearn.metrics.pairwise import euclidean_distances


def word2vec_embedding(reviews,size,min_count):
    ### build word embedding for each reviews
    word2vec_model = Word2Vec(reviews, size=size, min_count=min_count)
    return word2vec_model

def tf_idf_vectorizer(reviews, word2vec_model):
    descriptorized_reviews = [' '.join(review) for review in reviews]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit(descriptorized_reviews)
    dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))
    wine_review_vectors = []
    for review in descriptorized_reviews:
        weighted_review_terms = []
        words = review.split(' ')
        for word in words:
            if word in dict_of_tfidf_weightings.keys():
                tfidf_weighting = dict_of_tfidf_weightings[word]
                word_vector = word2vec_model.wv.get_vector(word).reshape(1, 100)
                weighted_word_vector = tfidf_weighting * word_vector
                weighted_review_terms.append(weighted_word_vector)
            else:
                continue
        try:
            review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
        except:
            review_vector = []
        wine_review_vectors.append(review_vector)
    return np.array(wine_review_vectors).reshape(-1,100)



