# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:54:35 2017

@author: ilja
"""

from __future__ import division
import math
import string
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model


# dataset import in Pandas
products = pd.read_csv('C:\\Users\\ilja.surikovs\\OneDrive - Accenture\\Machine Learning course\\Classification course\\week 1 task\\amazon_baby.csv') #, dtype = {'name':str,'review':str,'rating':int})

#function to remove punctuation
def remove_punctuation(text):
    if type(text)==str:
        import string
        return text.translate(str.maketrans('','',string.punctuation)) 

##function to create a dictionary of words - not needed with tf-idf
#def group_words(text_string):
#    if type(text_string)==str:
#        text_list = text_string.lower().split(' ')
#    #    print('text_list: ', text_list)
#    #    print('length of text_list: ', len(text_list))
#        text_list_unique = list(set(text_list))
#        if '' in text_list_unique:
#            text_list_unique.remove('')
#    #    print('text_list_unique: ', text_list_unique)
#        text_dict = {}
#        for word in text_list_unique:
#            text_dict[word] = text_string.count(word)
#        return text_dict
#    else:
#        return {}

#cleaning data
products['review'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products = products[products['review'].isnull()==False] # removing NaN values

#assigning sentiment
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

#splitting data
train_data, test_data = train_test_split(products, test_size=0.2)
print(len(train_data))
print(len(test_data))
print(len(products))

#creating matrix of all words an their counts from all train documents
my_vectorizer = CountVectorizer()
X_train_counts = my_vectorizer.fit_transform(train_data['review'])

      # checking vectorizer
            #print(train_data['review'].iloc[1:2])
            #test_count_vector = CountVectorizer().fit_transform(train_data['review'].iloc[1:2])
            #test_count_vector2 = CountVectorizer().fit_transform(['hello world','this is a a a text'])
            #test_count_vector2 = CountVectorizer().fit_transform(['a book and a pen volcano'])
            #print(test_count_vector2)
            #my_vectorizer3 = CountVectorizer()
            #test_count_vector3 = my_vectorizer.fit_transform(['this is a text and this is the book abrakadabragodzilla three three three','second list three'])
            #print(test_count_vector3)
            #my_vectorizer.get_feature_names()
            #my_vectorizer.get_params()
            #my_vectorizer.get_stop_words()

#X_train_counts.shape #to check the sparse matrix size

#performing tf-idfs transformations
my_tfidf_transfomer = TfidfTransformer()
X_train_tfidf = my_tfidf_transfomer.fit_transform(X_train_counts)
#X_train_tfidf.shape #to check the sparse matrix size

## defining Y and X
sentiment_series = train_data['sentiment']
sentiment_np = np.array(sentiment_series)

##running regressions
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train_tfidf, sentiment_np)
score = log_reg.score(X_train_tfidf, sentiment_np)
predictions = log_reg.predict(X_train_tfidf)
prob_predictions = log_reg.predict_proba(X_train_tfidf)

##showing regression results
print('regression coefficients: ', str(log_reg.coef_))
print('intercept: ', log_reg.intercept_)

print('my predictions: \n', predictions) 
print('actual sentiments: \n', sentiment_np) 
print('\n actual sentiments: mean=', np.mean(sentiment_np), ', median=', np.median(sentiment_np), ', min=', np.min(sentiment_np), ', max=',  np.max(sentiment_np))
print('\n predictions: mean=', np.mean(predictions), ', median=', np.median(predictions), ', min=', np.min(predictions), ', max=',np.max(predictions))

review_count = len(sentiment_np)
print('total number of sentiments=', review_count)
print('\nscore / mean accuracy (as given by package) = ',score)
print('accuracy (calculated manually) =', len(sentiment_np[sentiment_np==predictions]) / review_count)
print('majority class accuracy (train)=', len(sentiment_np[sentiment_np==1]) / review_count)

#forecasting with test data
X_test_counts = my_vectorizer.transform(test_data['review']) #need to call 'transform' on transformers instead of 'fit_transfor' since they have already been fit to the training set
X_test_tfidf = my_tfidf_transfomer.fit_transform(X_test_counts) ### !!!!!!!!!!!!!!! ???????????????? !!!!!!!!!!!!!!!
predictions_test = log_reg.predict(X_test_tfidf)
prob_predictions_test = log_reg.predict_proba(X_test_tfidf)


sentiment_series_test = test_data['sentiment']
sentiment_np_test = np.array(sentiment_series_test)
review_count_test = len(sentiment_np_test)
print('total number of sentiments_test=', review_count_test)
#print('\nscore / mean accuracy (as given by package) = ',score)
print('accuracy (calculated manually) =', len(sentiment_np_test[sentiment_np_test==predictions_test]) / review_count_test)
print('majority class accuracy (test)=', len(sentiment_np_test[sentiment_np_test==1]) / review_count_test)


