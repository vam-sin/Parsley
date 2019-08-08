# Libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
vectorizer = TfidfVectorizer()
import nltk
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
train = pd.read_csv('../dataset/train2.tsv', delimiter='\t', encoding='utf-8')
# print(train.shape) # 10239 statements with 16 columns
# for col in train.columns:
#     print(col)
# Required columns: 2(label), 3(statement), 15(justification)

# Test
test = pd.read_csv('../dataset/test2.tsv', delimiter='\t', encoding='utf-8')
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




# LSTM Model Model
model = Sequential()
model.add(Embedding(500, 32, input_length=500))
model.add(LSTM(32))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(padded_docs, y_train_binary,validation_data=(padded_docs_test,y_test_binary), epochs=10, verbose=1)

# Test data details
# loss: 0.6858 - acc: 0.5618 - val_loss: 0.6855 - val_acc: 0.5632
