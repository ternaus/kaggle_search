from __future__ import division
__author__ = 'Vladimir Iglovikov'

import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from nolearn import lasagne
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid
from nolearn.lasagne import NeuralNet
import numpy as np
import theano
from sklearn.metrics import roc_auc_score
import os
import time

from sklearn import cross_validation
from sklearn import metrics

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search


def clean_words(raw_txt):
    review_text = BeautifulSoup(raw_txt).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    return (' '. join(meaningful_words))

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)



def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

train = pd.read_csv('../data/train.csv')
train_query_clean = train['query'].apply(clean_words, 1).values
product_title_clean = train['product_title'].apply(clean_words, 1).values
train['product_description'].fillna(' ', inplace=True)
product_description_clean = train['product_description'].apply(clean_words, 1).values

train_qtd = np.concatenate([train_query_clean,
                      product_title_clean,
                      product_description_clean])

vectorizer = CountVectorizer(analyzer='word',
                            tokenizer=None,
                            preprocessor=None,
                            max_features=200
                            )
print 'fit vectorizer'
vectorizer.fit(train_qtd)

query_new = vectorizer.transform(train_query_clean)
title_new = vectorizer.transform(product_title_clean)
product_new = vectorizer.transform(product_description_clean)


train_new = np.concatenate((query_new.toarray(),
                            title_new.toarray(),
                            product_new.toarray()), 1)
y = train.median_relevance.values

random_state = 42


layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense1', DenseLayer),
           # ('dropout2', DropoutLayer),
           # ('dense2', DenseLayer),
           ('output', DenseLayer),
           ]
num_units = 10

num_classes = len(train.median_relevance.unique())
num_features = train_new.shape[1]

clf = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=num_units,
                 dropout1_p=0.5,
                 dense1_num_units=num_units,
                 # dropout2_p=0.5,
                 # dense2_num_units=num_units,
                 output_num_units=4,
                # output_num_units=1,
                 output_nonlinearity=softmax,
                #  output_nonlinearity=rectify(),
                # output_nonlinearity=sigmoid,

                 update=nesterov_momentum,
                 # update_learning_rate=0.001,
                 # update_momentum=0.9,
                 update_momentum=theano.shared(float32(0.9)),
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=1000,
                 update_learning_rate=theano.shared(float32(0.03)),
                 # objective_loss_function= binary_crossentropy,
                 on_epoch_finished=[
                    AdaptiveVariable('update_learning_rate', start=0.0001, stop=0.000001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(patience=100),
                ])

# clf = RandomForestClassifier(n_jobs=3, n_estimators=10, random_state=random_state)

kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
scores = cross_validation.cross_val_score(clf,
                                          train_new.astype(np.float32),
                                          y.astype(np.int32),
                                          cv=skf,
                                          scoring='log_loss'
                                          # scoring=kappa_scorer
                                          )
print np.mean(scores), np.std(scores)