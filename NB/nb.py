# Libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
# Required columns: 2(label), 3(statement), 15(justification)

# Test
test = pd.read_csv('../dataset/test2.tsv', delimiter='\t', encoding='utf-8')
x_test = test.iloc[:,[3,15]]
x_test = np.asarray(x_test)
x_test = x_test.tolist()
X_test = []
for d in x_test:
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
for d in x_train:
    d = str(d[0])
    d = d + str(d[1])
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

# TfIDf
tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(list(X)+list(X_test))
x_train_tfv = tfv.transform(X)
x_test_tfv = tfv.transform(X_test)

# CountVectorizer
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
ctv.fit(list(X) + list(X_test))
xtrain_ctv =  ctv.transform(X)
xtest_ctv = ctv.transform(X_test)

# NB Model
nbmodel = MultinomialNB()
nbmodel.fit(xtrain_ctv, y_train_binary)
predictions = nbmodel.predict(xtest_ctv)
print(accuracy_score(y_test_binary,predictions))

# Test data details
# 0.6058451816745656 (With TFIDF)
# 0.6003159557661928 (With CountVectorizer)
