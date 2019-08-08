# Ensemble of LR, LSTM, XGB Models
# Tasks Left
# Better Text to Number conversion
# Add the justification column.
# Add the features column.
# get the accuracy values for the six way classifier also.
# Libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from sklearn.linear_model import LogisticRegression
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import xgboost as xgb
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, LSTM, Flatten
from sklearn.ensemble import VotingClassifier
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


# Import Train dataset
train = pd.read_csv('dataset/train2.tsv', delimiter='\t', encoding='utf-8')
# print(train.shape) # 10239 statements with 16 columns
# for col in train.columns:
#     print(col)
# Required columns: 2(label), 3(statement), 15(justification)

# Test
test = pd.read_csv('dataset/test2.tsv', delimiter='\t', encoding='utf-8')
x_test = test.iloc[:,3:4]
x_test = np.asarray(x_test)
x_test = x_test.tolist()
X_test = []
for d in x_test:
    d = str(d[0])
    X_test.append(d)

y_test = test.iloc[:,2:3]
y_test = np.asarray(y_test)

# X and Y (Train)
# x_train = train.iloc[:,[3,15]] (statement & justification)
x_train = train.iloc[:, 3:4]
x_train = np.asarray(x_train)
x_train = x_train.tolist()
X = []
for d in x_train:
    d = str(d[0])
    X.append(d)
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
# Input Word Embeddings
vocab_size = 500
encoded_docs = [one_hot(d, vocab_size) for d in X]
max_length = 500
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
print(padded_docs.shape)
# Test
vocab_size = 500
encoded_docs_test = [one_hot(d, vocab_size) for d in X_test]
max_length = 500
padded_docs_test = pad_sequences(encoded_docs_test, maxlen = max_length, padding = 'post')

# print(x_train[0][0])
# print(text_process(x_train[0][0]))
# for i in range(len(x_train)):
#     for j in range(len(x_train[i])):
#         x_train[i][j] = text_process(x_train[i][j])
# TfIDf
# Word2Vec
# GloVe


# Logistic Regression Model
print("Logistic Regression")
logmodel = LogisticRegression()
logmodel.fit(padded_docs, y_train_binary)
# predictions = logmodel.predict(padded_docs_test)
# print(accuracy_score(y_test_binary,predictions))

# XGB
print("XGB")
xgb_model = xgb.XGBClassifier(random_state = 1, learning_rate = 0.01)
xgb_model.fit(padded_docs, y_train_binary)

# LSTM
print("LSTM")
model = Sequential()
model.add(Embedding(500, 32, input_length=500))
model.add(LSTM(32))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())
# model.fit(padded_docs, y_train_binary,validation_data=(padded_docs_test,y_test_binary), epochs=1, verbose=1)
# model.save('lstm.h5')
model = load_model('lstm.h5')

# Ensemble
print("Ensemble")
# estimators = [('logmodel', logmodel), ('xgb_model', xgb_model), ('model', model)]
# ensemble = VotingClassifier(estimators, voting='hard')
# ensemble.fit(padded_docs, y_train_binary)
# ensemble.score(padded_docs, y_test_binary)
pred1 = logmodel.predict(padded_docs_test)
pred2 = xgb_model.predict(padded_docs_test)
pred3 = model.predict(padded_docs_test)
pred3_bin = []
for i in pred3:
    if i >= 0.5:
        pred3_bin.append(1)
    else:
        pred3_bin.append(0)
pred3_bin = np.asarray(pred3_bin)
print(pred3_bin.shape, y_test_binary.shape)
# print(pred1, pred2, pred3_bin)

final_pred = []
for i in range(0, len(y_test_binary)):
    # print(pred1[i], pred2[i], pred3_bin[i])
    data = [pred1[i], pred2[i], pred3_bin[i]]
    print(data)
    val = mode(data)
    final_pred.append(val)

final_pred = np.asarray(final_pred)
print(accuracy_score(final_pred, y_test_binary))

# Test data details
# 0.5655608214849921
