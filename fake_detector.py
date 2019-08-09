# Ensemble of LR, DL, XGB, Adaboost and NaiveBayes Models
# Tasks Left
# Better Text to Number conversion (Done)
# Ensemble Model creation (Stacking, take all the outputs from the base learners and make an nn with a dense layer which
# learns from them to give the output)
# Plots in 2D
# Add the features column.
# get the accuracy values for the six way classifier also.
# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, SpatialDropout1D, GRU, Embedding
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence, text
from sklearn.model_selection import StratifiedKFold

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
train = pd.read_csv('dataset/train2.tsv', delimiter='\t', encoding='utf-8')
# print(train.shape) # 10239 statements with 16 columns
# for col in train.columns:
#     print(col)
# Required columns: 2(label), 3(statement), 15(justification)

# Test
test = pd.read_csv('dataset/test2.tsv', delimiter='\t', encoding='utf-8')
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
for d in x_train:
    result = analyzer.polarity_scores(str(d[0]))
    # print(str(d[0]))
    xtrain_sent_1.append(result['compound'])
    result = analyzer.polarity_scores(str(d[1]))
    xtrain_sent_2.append(result['compound'])
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
# Input Word Embeddings
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
ctv.fit(list(X) + list(X_test))
xtrain_ctv =  ctv.transform(X)
xtest_ctv = ctv.transform(X_test)

# Logistic Regression Model
print("Logistic Regression")
logmodel = LogisticRegression()
logmodel.fit(xtrain_ctv, y_train_binary)
# predictions = logmodel.predict(padded_docs_test)
# print(accuracy_score(y_test_binary,predictions))

# NaiveBayes
print("NB")
nbmodel = MultinomialNB()
nbmodel.fit(xtrain_ctv, y_train_binary)

# XGB
# print("XGB")
# xgb_model = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
#                         subsample=0.8, nthread=10, learning_rate=0.1)
# xgb_model.fit(xtrain_ctv, y_train_binary)

# Adaboost
# print("Adaboost")
# clf = AdaBoostClassifier(n_estimators=100)
# clf.fit(xtrain_ctv, y_train_binary)
# y_pred = clf.predict(xtest_ctv)
# print(accuracy_score(y_test_binary, y_pred))
# 0.590047393364929 (Accuracy)

# Deep Learning Model
print("GRU")
# Glove
embeddings_index = {}
f = open('glove.42B.300d.txt', encoding='utf8')
for line in tqdm(f):
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

xtrain_glove = [sent2vec(x) for x in tqdm(X)]
xtest_glove = [sent2vec(x) for x in tqdm(X_test)]

# Scaling
scl = preprocessing.StandardScaler()
xtrain_glove_scl = scl.fit_transform(xtrain_glove)
xtest_glove_scl = scl.transform(xtest_glove)

token = text.Tokenizer(num_words=None)
max_len = 300

token.fit_on_texts(list(X) + list(X_test))
xtrain_seq = token.texts_to_sequences(X)
xvalid_seq = token.texts_to_sequences(X_test)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

nn = Sequential()
nn.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
nn.add(SpatialDropout1D(0.3))
nn.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
nn.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))
nn.add(Dense(1024, activation='relu'))
nn.add(Dropout(0.8))
nn.add(Dense(1024, activation='relu'))
nn.add(Dropout(0.8))
nn.add(Dense(1, activation='sigmoid'))
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
nn = load_model('gru.h5')

# Ensemble
print("Ensemble")
# estimators = [('logmodel', logmodel), ('xgb_model', xgb_model), ('model', model)]
# ensemble = VotingClassifier(estimators, voting='hard')
# ensemble.fit(padded_docs, y_train_binary)
# ensemble.score(padded_docs, y_test_binary)
# pred1 = logmodel.predict(xtrain_ctv)
# # pred2 = xgb_model.predict(xtrain_ctv)
# pred3 = nbmodel.predict(xtrain_ctv)
# pred4 = nn.predict(xtrain_pad)
pred1_test = logmodel.predict(xtest_ctv)
# pred2_test = xgb_model.predict(xtest_ctv)
pred3_test = nbmodel.predict(xtest_ctv)
pred4_test = nn.predict(xtest_pad)

pred4_bin = []
for i in pred4_test:
    if i >= 0.5:
        pred4_bin.append(1)
    else:
        pred4_bin.append(0)
pred4_bin = np.asarray(pred4_bin)

# next_pred = []
# for i in range(0, len(y_train_binary)):
#     # print(pred1[i], pred2[i], pred3_bin[i])
#     data = [pred1[i], pred3[i], pred4[i]]
#     # print(data)
#     val = mode(data)
#     next_pred.append(data)
#
# next_pred = np.asarray(next_pred)

next_pred_test = []
for i in range(0, len(y_test_binary)):
    # print(pred1[i], pred2[i], pred3_bin[i])
    data = [pred1_test[i], pred3_test[i], pred4_bin[i]]
    # print(data)
    val = mode(data)
    next_pred_test.append(val)

next_pred_test = np.asarray(next_pred_test)
print(accuracy_score(next_pred_test, y_test_binary))


# Stacked Model
# model = Sequential()
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# checkpoint = ModelCheckpoint('final_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# model.fit(next_pred, y_train_binary,validation_data=(next_pred_test, y_test_binary), batch_size=64, epochs=50, verbose=1, callbacks=callbacks_list)


# final_pred = np.asarray(final_pred)
# print(accuracy_score(final_pred, y_test_binary))

# Test data details
# (Stacked Model)
# ALL: 0.62243
# Without NaiveBayes: 0.62164 (Needed)
# Without XGB: 0.62322 (Nope)
# Without LR: 0.59874 (Needed)
# Mode Counting without sentiment
# 0.6240126382306477
