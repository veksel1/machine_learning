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

