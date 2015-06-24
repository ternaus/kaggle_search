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

clf = RandomForestClassifier(n_jobs=3, n_estimators=10, random_state=random_state)

kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)
skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
scores = cross_validation.cross_val_score(clf, train_new, y, cv=skf, scoring=kappa_scorer)
print np.mean(scores), np.std(scores)