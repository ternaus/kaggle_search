from __future__ import division
__author__ = 'Vladimir Iglovikov'

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

import sys
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation
from nltk.corpus import stopwords
import sys
from sklearn import metrics
import numpy as np
sys.path += ['src', '/home/vladimir/compile/xgboost/wrapper']

import xgboost as xgb
from kappa_squared import quadratic_weighted_kappa
from bs4 import BeautifulSoup
import re
from sklearn.decomposition import TruncatedSVD
from pylab import *
import seaborn as sns
from nltk.stem.porter import *
from sklearn.feature_extraction import text

sw=[]
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)

# import gl_wrapper
# import graphlab as gl

'''
Here I will try to work on a problem in a way that I would expect to be correct

Meaning important features:
[1] Cos between query and product title. Both cleaned and not cleaned.
'''

train = pd.read_csv('../data/train.csv').fillna("")
test = pd. read_csv('../data/test.csv').fillna("")

# train['product_description'].fillna(' ', inplace=True)
# test['product_description'].fillna(' ', inplace=True)

train['query'] = train['query'].apply(lambda x: BeautifulSoup(x).get_text(), 1).apply(lambda x: x.lower(), 1)
test['query'] = test['query'].apply(lambda x: BeautifulSoup(x).get_text(), 1).apply(lambda x: x.lower(), 1)

train['product_title'] = (train['product_title']
                          .apply(lambda x: BeautifulSoup( re.sub(r'^https?:\/\/.*[\r\n]*', '', x)).get_text(), 1)
                          .apply(lambda x: x.lower(), 1)
                          )
test['product_title'] = (test['product_title']
                         .apply(lambda x: BeautifulSoup( re.sub(r'^https?:\/\/.*[\r\n]*', '', x)).get_text(), 1)
                         .apply(lambda x: x.lower(), 1)
                         )

train['product_description'] = (train['product_description']
                                .apply(lambda x: BeautifulSoup( re.sub(r'^https?:\/\/.*[\r\n]*', '', x)).get_text(), 1)
                                .apply(lambda x: x.lower(), 1)
                         )

test['product_description'] = (test['product_description']
                               .apply(lambda x: BeautifulSoup( re.sub(r'^https?:\/\/.*[\r\n]*', '', x)).get_text(), 1)
                               .apply(lambda x: x.lower(), 1)
                               )

train.to_pickle('../data/train1')
test.to_pickle('../data/test1')

sys.exit()

# train = pd.read_pickle('../data/train1')
# test = pd.read_pickle('../data/test1')

stemmer = PorterStemmer()
traindata = list((train
  .apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1)
  .apply(lambda x: re.sub("[^a-zA-Z0-9]"," ", x))
  .apply(lambda x: x.strip())
  .apply(lambda x: (" ").join([stemmer.stem(z) for z in x.split(" ")]))))

testdata = list((test.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1)
.apply(lambda x: re.sub("[^a-zA-Z0-9]"," ", x))))
  .apply(lambda x: re.sub("[^a-zA-Z0-9]"," ", x))
  .apply(lambda x: x.strip())
  .apply(lambda x: (" ").join([stemmer.stem(z) for z in x.split(" ")]))))

)

# traindata = list(train.apply(lambda x: '%s %s' % (x['query'], x['product_title']), axis=1).apply(lambda x: re.sub("[^a-zA-Z0-9]"," ", x)))
# testdata = list(test.apply(lambda x: '%s %s' % (x['query'], x['product_title']), axis=1).apply(lambda x: re.sub("[^a-zA-Z0-9]"," ", x)))


corpus = traindata + testdata
tfv = TfidfVectorizer(min_df=5,
                      max_df=500,
                          max_features=12000,
                                   # max_features=None,
                          strip_accents='unicode',
                          analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 2),
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1,
                          # stop_words = stop_words
                          stop_words=list(set(stopwords.words("english")))
                          )
print
print 'fitting corpus'
tfv.fit(corpus)

print 'create weight'
a = train.groupby('query')['relevance_variance'].mean()
a = pd.DataFrame(a)
a.reset_index(inplace=True)
a.rename(columns={'relevance_variance':'weight'}, inplace=True)
b = train.merge(a, on='query')

# print tfv.vocabulary
# sys.exit()
print 'transforming columns'
query_new = tfv.transform(train['query']).toarray()
title_new = tfv.transform(train['product_title']).toarray()
description_new = tfv.transform(train['product_description']).toarray()

result = []
for i in range(len(query_new)):
    result += [(cosine(query_new[i], title_new[i]),
                cosine(query_new[i], description_new[i]),
                cosine(title_new[i], description_new[i])
               )
              ]


new_train = pd.DataFrame(result)
new_train.columns = ['cosine_qt', 'cosine_qd', 'cosine_td']
# new_train.columns = ['cosine_qt', 'cosine_qd']
new_train.cosine_qt.fillna(0, inplace=True)
new_train.cosine_qd.fillna(0, inplace=True)
new_train.cosine_td.fillna(0, inplace=True)
new_train['weight'] = b['weight']

print 'transforming train'
qt = tfv.transform(train.apply(lambda x: '%s %s' % (x['query'], x['product_title']), axis=1).values)

kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

random_state = 42

svd = TruncatedSVD(n_components=200)

qt_transformed = svd.fit_transform(qt)

X = np.concatenate((new_train.values, qt_transformed), 1)
# X = qt_transformed
y = train['median_relevance'].values

ind = 2

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
# clf = xgb.XGBClassifier(silent=False)
# clf = gl_wrapper.BoostedTreesClassifier()


params = {'max_iterations': 100,
          # 'max_depth': 5,
          # 'min_loss_reduction': 1,
          # 'step_size': 0.1,
          # 'row_subsample': 0.8,
          # 'column_subsample': 0.7,
           }

if ind == 1:
  print 'estimating'
  X = pd.DataFrame(X)
  features = X.columns
  X['target'] = y
  params['target'] = 'target'
  sf_train, sf_test = gl.SFrame(X).random_split(0.7, seed=5)
  model = gl.boosted_trees_classifier.create(sf_train, validation_set=sf_test, **params)
elif ind == 2:
  print 'estimating'
  skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
  # scores = cross_validation.cross_val_score(clf, X, y, cv=skf)
  scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring=kappa_scorer)
  print np.mean(scores), np.std(scores)
elif ind == 3:
  query_new_test = tfv.transform(test['query']).toarray()
  title_new_test = tfv.transform(test['product_title']).toarray()
  description_new_test = tfv.transform(test['product_description']).toarray()


  result_test = []
  for i in range(len(query_new_test)):
    result_test += [(cosine(query_new_test[i], title_new_test[i]),
                cosine(query_new_test[i], description_new_test[i]),
                cosine(title_new_test[i], description_new_test[i])
               )
              ]

  new_test = pd.DataFrame(result_test)
  new_test.columns = ['cosine_qt', 'cosine_qd', 'cosine_td']
  new_test.cosine_qt.fillna(0, inplace=True)
  new_test.cosine_qd.fillna(0, inplace=True)
  new_test.cosine_td.fillna(0, inplace=True)
  new_test['weight'] = b['weight']
  print 'transforming test' 


  qt_test = tfv.transform(test.apply(lambda x: '%s %s' % (x['query'], x['product_title']), axis=1).values)
  qt_test_transformed = svd.transform(qt_test)

  X_test = np.concatenate((new_test.values, qt_test_transformed), 1)

  clf.fit(X, y)

  prediction = clf.predict(X_test)
  submission = pd.DataFrame()
  submission['id'] = test['id']
  submission['prediction'] = prediction
  submission.to_csv('prediction/RF.csv', index=False)