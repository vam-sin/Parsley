# Libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
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
x_test_st = test.iloc[:,3:4]
x_test_st = np.asarray(x_test_st)
x_test_st = x_test_st.tolist()
X_test = []
for d in x_test_st:
    d = str(d[0])
    X_test.append(d)
x_test_ju = test.iloc[:, 15:16]
x_test_ju = np.asarray(x_test_ju)
x_test_ju = x_test_ju.tolist()
X_test_ju = []
for d in x_test_ju:
    d = str(d[0])
    X_test_ju.append(d)
y_test = test.iloc[:,2:3]
y_test = np.asarray(y_test)

# X and Y (Train)
# x_train = train.iloc[:,[3,15]] (statement & justification)
x_train_st = train.iloc[:, 3:4]
x_train_st = np.asarray(x_train_st)
x_train_st = x_train_st.tolist()
X = []
for d in x_train_st:
    d = str(d[0])
    X.append(d)
x_train_ju = train.iloc[:, 15:16]
x_train_ju = np.asarray(x_train_ju)
x_train_ju = x_train_ju.tolist()
X_ju = []
for d in x_train_ju:
    d = str(d[0])
    X_ju.append(d)
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
vocab_size = 100
encoded_docs = [one_hot(d, vocab_size) for d in X]
max_length = 100
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
print(padded_docs)
padded_docs = np.asarray(padded_docs)
encoded_docs_ju = [one_hot(d, 300) for d in X_ju]
padded_docs_ju = pad_sequences(encoded_docs_ju, maxlen = 300, padding = 'post')
print(padded_docs_ju)
padded_docs_ju = np.asarray(padded_docs_ju)
X_Tr = np.concatenate((padded_docs, padded_docs_ju), axis=1)
print(X_Tr.shape)
# Test
vocab_size = 100
encoded_docs_test = [one_hot(d, vocab_size) for d in X_test]
max_length = 100
padded_docs_test = pad_sequences(encoded_docs_test, maxlen = max_length, padding = 'post')
padded_docs_test = np.asarray(padded_docs_test)
encoded_docs_ju_test = [one_hot(d, 300) for d in X_test_ju]
padded_docs_ju_test = pad_sequences(encoded_docs_ju_test, maxlen = 300, padding = 'post')
print(padded_docs_ju)
padded_docs_ju_test = np.asarray(padded_docs_ju_test)
X_Te = np.concatenate((padded_docs_test, padded_docs_ju_test), axis=1)
# Logistic Regression Model
logmodel = LogisticRegression()
logmodel.fit(X_Tr, y_train_binary)
predictions = logmodel.predict(X_Te)
print(accuracy_score(y_test_binary,predictions))

# Test data details
# 0.54739336492891
