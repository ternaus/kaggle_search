from __future__ import division
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from sklearn import cross_validation
from nltk.corpus import stopwords
from kappa_squared import quadratic_weighted_kappa
from sklearn import metrics
from scipy.spatial.distance import cosine
from nltk.stem.porter import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import time

stemmer = PorterStemmer()

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

def review_to_wordlist( review, remove_stopwords=False ):
    fix_dict = {'rechargable': 'rechargeable',
  'fragance': 'fragrance',
  'extenal': 'external',
  'hardisk': 'harddisk',
  'refrigirator': 'refrigerator',
  'qualtiy': 'quality',
  'ilalian': 'italian',
  'offcially': 'officially',
  'batterry': 'battery',
  'smarphone': 'smartphone',
  'wless': 'wireless',
  'kboard': 'keyboard'}


    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    # review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)
    review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    result = []
    for word in words:
      if word in fix_dict:
        result += [stemmer.stem(fix_dict[word])]
      else:
        result += [stemmer.stem(word)]
    #
    # 5. Return a list of words
    return(result)

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences
# sentences = set()  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in set(train["query"].values):
    sentences += review_to_sentences(review.decode('utf8', 'ignore'), tokenizer)
    # sentences.add(review_to_sentences(review.decode('utf8', 'ignore'), tokenizer))

# for review in test["query"]:
#     sentences += review_to_sentences(review.decode('utf8', 'ignore'), tokenizer)
#     # sentences.add(review_to_sentences(review.decode('utf8', 'ignore'), tokenizer))

for review in train["product_title"]:
    sentences += review_to_sentences(review.decode('utf8', 'ignore'), tokenizer)
    # sentences.add(review_to_sentences(review.decode('utf8', 'ignore'), tokenizer))

# for review in test["product_title"]:
#     sentences += review_to_sentences(review.decode('utf8', 'ignore'), tokenizer)

# sentences = set(sentences)
# train['product_description'].fillna(' ', inplace=True)

# for review in train["product_description"]:
#     sentences += review_to_sentences(review.decode('utf8', 'ignore'), tokenizer)

    # Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
# min_word_count = 5   # Minimum word count                        
min_word_count = 1   # Minimum word count                        
# num_workers = 40       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
# model_name = "models/300features_40minwords_5context"
# model.save(model_name)

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = int(word_vectors.shape[0] / 5)

random_state=42

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters, 
  n_jobs = -1, 
  precompute_distances=True,
  random_state=random_state,
  verbose=1)
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.index2word, idx ))

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

remove_stopwords = False



clean_train_query = []
for review in train["query"]:
    clean_train_query.append( review_to_wordlist(review.decode('utf8', 'ignore'), remove_stopwords=remove_stopwords))

clean_train_title = []
for review in train["product_title"]:
    clean_train_title.append( review_to_wordlist(review.decode('utf8', 'ignore'), remove_stopwords=remove_stopwords ))


# Pre-allocate an array for the training set bags of centroids (for speed)
train_query_centroids = np.zeros( (train["query"].size, num_clusters), dtype="float32" )

train_title_centroids = np.zeros( (train["query"].size, num_clusters), dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_query:
    train_query_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_title:
    train_title_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1


print 'create weight'
a = train.groupby('query')['relevance_variance'].mean()
a = pd.DataFrame(a)
a.reset_index(inplace=True)
a.rename(columns={'relevance_variance':'weight'}, inplace=True)
b = train.merge(a, on='query')

result = []
for i in range(train.shape[0]):
    result += [(cosine(train_query_centroids[i], train_title_centroids[i]),
                # cosine(query_new[i], description_new[i]),
                # cosine(title_new[i], description_new[i])
               )
              ]


new_train = pd.DataFrame(result)
# new_train.columns = ['cosine_qt', 'cosine_qd', 'cosine_td']
# new_train.columns = ['cosine_qt', 'cosine_qd']
new_train.columns = ['cosine_qt']
new_train.cosine_qt.fillna(0, inplace=True)
# new_train.cosine_qd.fillna(0, inplace=True)
# new_train.cosine_td.fillna(0, inplace=True)
new_train['weight'] = b['weight']



y = train['median_relevance'].values



X = pd.conatenate((new_train.values, train_query_centroids, train_title_centroids), 1)

random_state = 42
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, n_jobs=3)

print 'estimating'
skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
# scores = cross_validation.cross_val_score(clf, X, y, cv=skf)
scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring=kappa_scorer, n_jobs=-1)
print np.mean(scores), np.std(scores)



# clean_train_title = []
# for review in train["product_title"]:
#     clean_train_title.append( review_to_wordlist(review.decode('utf8', 'ignore'), remove_stopwords=remove_stopwords ))

# title_new = getAvgFeatureVecs( clean_train_title, model, num_features)

# kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

# random_state = 42

# print 'create weight'
# a = train.groupby('query')['relevance_variance'].mean()
# a = pd.DataFrame(a)
# a.reset_index(inplace=True)
# a.rename(columns={'relevance_variance':'weight'}, inplace=True)
# b = train.merge(a, on='query')

# result = []
# for i in range(train.shape[0]):
#     result += [(cosine(query_new[i], title_new[i]),
#                 # cosine(query_new[i], description_new[i]),
#                 # cosine(title_new[i], description_new[i])
#                )
#               ]


# new_train = pd.DataFrame(result)
# # new_train.columns = ['cosine_qt', 'cosine_qd', 'cosine_td']
# # new_train.columns = ['cosine_qt', 'cosine_qd']
# new_train.columns = ['cosine_qt']
# new_train.cosine_qt.fillna(0, inplace=True)
# # new_train.cosine_qd.fillna(0, inplace=True)
# # new_train.cosine_td.fillna(0, inplace=True)
# new_train['weight'] = b['weight']


# # X = np.concatenate((new_train.values, qt_transformed), 1)



# # X = np.concatenate([trainDataVecs_query, trainDataVecs_title], 1)
# # X = trainDataVecs_query

# X = np.concatenate((new_train.values, query_new, title_new), 1)


# # scaler = StandardScaler()
# # X = scaler.fit_transform(X)

# # ind = 3

# # clf = SVC(C=10.0, 
# #   kernel='rbf', 
# #   degree=3, 
# #   gamma=0.0, 
# #   coef0=0.0, 
# #   shrinking=True, 
# #   # probability=False, 
# #   probability=True, 
# #   tol=0.001, 
# #   cache_size=4096, 
# #   class_weight=None, 
# #   verbose=False, 
# #   max_iter=-1, 
# #   random_state=42)


# y = train['median_relevance'].values

# # Fit a random forest to the training data, using 100 trees
# from sklearn.ensemble import RandomForestClassifier
# # clf = RandomForestClassifier(n_estimators = 100, n_jobs=3)

# print 'estimating'
# skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
# # scores = cross_validation.cross_val_score(clf, X, y, cv=skf)
# scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring=kappa_scorer, n_jobs=-1)
# print np.mean(scores), np.std(scores)


# # print "Creating average feature vecs for test reviews"
# # clean_test_reviews = []
# # for review in test["review"]:
# #     clean_test_reviews.append( review_to_wordlist( review, \
# #         remove_stopwords=True ))

# # testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )