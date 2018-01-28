# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:54:35 2017

@author: ilja.surikovs
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
from sklearn import tree

import matplotlib.pyplot as plt

# dataset import in Pandas
loans = pd.read_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\ml-classification\\week3\\lending-club-data.csv') #, dtype = {'name':str,'review':str,'rating':int})

# observing dataset
print(loans['id'])
print(loans['id'][0:5])
print(loans[0:5])
print(list(loans)) #show column names

#showing loans by 'grade' and by 'home_ownership'
loans['grade'].hist(bins=7)
loans['home_ownership'].hist(bins=4)

#creating column 'safe_loans'
len(loans['bad_loans'])
len(loans[loans['bad_loans']==0])
len(loans[loans['bad_loans']==1])
loans['safe_loans']=loans['bad_loans']+1
loans['safe_loans'][loans['safe_loans']==2] =-1
loans['safe_loans'].hist(bins=2)

#defining features and the target
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'

# Extract the feature columns and target column
loans = loans[features + [target]]

#splitting to 'safe' and 'risky' datasets
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))

#making datasets balanced, merging them, adding dummies for categorical variables 
#and splitting to train_data and validation_data
riskyToSafeRatio = len(risky_loans_raw)/len(safe_loans_raw)
risky_loans = risky_loans_raw
safe_loans=safe_loans_raw.sample(frac=riskyToSafeRatio,random_state=1) 
loans_data = risky_loans.append(safe_loans)
loans_data_with_dummies = pd.get_dummies(loans_data)

train_data, validation_data = train_test_split(loans_data_with_dummies, test_size=0.2, random_state=1)


#training the decision tree classifier
decision_tree_model = tree.DecisionTreeClassifier()
small_model = tree.DecisionTreeClassifier(max_depth=2)
#np_train_data = np.array(train_data)
#np_validation_data = np.array(validation_data)
#np_target = np.array(train_data[target])

train_data_wo_target = train_data[train_data.columns.difference([target])]
decision_tree_model.fit(train_data_wo_target,train_data[target])
small_model.fit(train_data_wo_target,train_data[target])

#getting samples for predictions
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data
sample_validation_data_wo_target = sample_validation_data[sample_validation_data.columns.difference([target])]

#making predictions

decision_tree_model.predict(sample_validation_data_wo_target)
small_model.predict(sample_validation_data_wo_target)

decision_tree_model.predict_proba(sample_validation_data_wo_target)
small_model.predict_proba(sample_validation_data_wo_target)

decision_tree_model.score(train_data_wo_target,train_data[target])
small_model.score(train_data_wo_target,train_data[target])

validation_data_wo_target = validation_data[validation_data.columns.difference([target])]

decision_tree_model.score(validation_data_wo_target,validation_data[target])
small_model.score(validation_data_wo_target,validation_data[target])

#finding optimal model
for depth in range(1,100):
    optimal_model = tree.DecisionTreeClassifier(max_depth=depth)
    optimal_model.fit(train_data_wo_target,train_data[target])
    print('depth =', depth)
    print('train accuracy:',optimal_model.score(train_data_wo_target,train_data[target]))
    print('valid accuracy:',optimal_model.score(validation_data_wo_target,validation_data[target]))
    print()

#majority class estimator
print('train majority class accuracy:', len(train_data[train_data[target]==+1])/len(train_data))
print('valid majority class accuracy:', len(validation_data[validation_data[target]==+1])/len(validation_data))




##running regressions
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train_tfidf, sentiment_np)






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


