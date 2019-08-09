# Libraries
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse.linalg import svds, eigs
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = stopwords.words('english')
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#Functions
# Convert to binary
def convert_to_bin(values):
    bin = []
    for label in values:
        if label=='true' or label=='half-true' or label=='mostly-true':
            bin.append('true')
        else:
            bin.append('false')

    return bin

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# Import Train dataset
train = pd.read_csv('../dataset/train2.tsv', delimiter='\t', encoding='utf-8')
# Required columns: 2(label), 3(statement), 15(justification)

# Test
test = pd.read_csv('../dataset/test2.tsv', delimiter='\t', encoding='utf-8')
x_test = test.iloc[:,[3,15]]
x_test = np.asarray(x_test)
x_test = x_test.tolist()
X_test = []
xtest_sent_1 = []
xtest_sent_2 = []
analyzer = SentimentIntensityAnalyzer()
for d in x_test:
    result = analyzer.polarity_scores(str(d[0]))
    # print(str(d[0]))
    xtest_sent_1.append(result['compound'])
    result = analyzer.polarity_scores(str(d[1]))
    xtest_sent_2.append(result['compound'])
    d = str(d[0])
    d = d + str(d[1])
    X_test.append(d)

y_test = test.iloc[:,2:3]
y_test = np.asarray(y_test)

# X and Y (Train)
# x_train = train.iloc[:,[3,15]] (statement & justification)
x_train = train.iloc[:, [3,15]]
x_train = np.asarray(x_train)
x_train = x_train.tolist()
X = []
xtrain_sent_1 = []
xtrain_sent_2 = []
analyzer = SentimentIntensityAnalyzer()
for d in x_train:
    result = analyzer.polarity_scores(str(d[0]))
    # print(str(d[0]))
    xtrain_sent_1.append(result['compound'])
    result = analyzer.polarity_scores(str(d[1]))
    xtrain_sent_2.append(result['compound'])
    d = str(d[0])
    d = d + str(d[1])
    X.append(d)

# xtrain_sent_1 = np.asarray(xtrain_sent_1)
# xtrain_sent_1 = np.reshape(xtrain_sent_1, (10239, 1))
# xtrain_sent_2 = np.asarray(xtrain_sent_2)
# xtrain_sent_2 = np.reshape(xtrain_sent_2, (10239, 1))
# print(X.shape, xtrain_sent_1.shape)
# print(X)
# x_train = np.asarray(x_train)
# print(x_train)
y_train = train.iloc[:,2:3]
y_train = np.asarray(y_train)

# Convert to Binary Classification problem
y_train_binary = convert_to_bin(y_train)
y_train_binary = np.asarray(y_train_binary)
# y_train_binary = np.reshape(y_train_binary, (10239, 1))
y_test_binary = convert_to_bin(y_test)
y_test_binary = np.asarray(y_test_binary)
# y_test_binary = np.reshape(y_test_binary, (1266, 1))

# Dataset Preprocessing
# Output (True:1, False:0)
le = preprocessing.LabelEncoder()
y_train_binary = le.fit_transform(y_train_binary)
y_test_binary = le.fit_transform(y_test_binary)

# TfIDf
# tfv = TfidfVectorizer(min_df=3,  max_features=None,
#             strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#             ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
#             stop_words = 'english')
# tfv.fit(list(X)+list(X_test))
# x_train_tfv = tfv.transform(X)
# x_test_tfv = tfv.transform(X_test)

# CountVectorizer
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
ctv.fit(list(X) + list(X_test))
xtrain_ctv =  ctv.transform(X)
xtrain_ctv = xtrain_ctv.astype(float)
xtest_ctv = ctv.transform(X_test)
# Word2Vec
# embeddings_index = {}
# f = open('../glove.42B.300d.txt', encoding='utf8')
# for line in tqdm(f):
#     values = line.split()
#     word = ''.join(values[:-300])
#     coefs = np.asarray(values[-300:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# xtrain_glove = [sent2vec(x) for x in tqdm(X)]
# xtest_glove = [sent2vec(x) for x in tqdm(X_test)]

# print(xtrain_ctv.shape)
# sent = np.concatenate((xtrain_sent_1, xtrain_sent_2), axis=1)
# print(sent.shape)
# xtrain_ctv = np.asarray(xtrain_ctv)
# xtrain_ctv = xtrain_ctv.astype('float64')
# x_train = np.concatenate((xtrain_ctv, sent), axis=1)
# print(x_train.shape)
#
# sent_test = np.concatenate((xtest_sent_1, xtest_sent_2), axis=1)
# sent_test = list(sent_test)
# xtest_ctv = list(xtest_ctv)
# x_test = np.append(xtest_ctv, sent_test, axis=1)
# x_test = np.asarray(x_test)

# Logistic Regression Model
logmodel = LogisticRegression()
logmodel.fit(xtrain_ctv, y_train_binary)
predictions = logmodel.predict(xtest_ctv)
print(accuracy_score(y_test_binary,predictions))

# Test data details
# 0.6216429699842022 (With CountVectorizer)
# 0.6058451816745656 (With GloVe)
