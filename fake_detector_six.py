# Ensemble of LR, DL, XGB, Adaboost and NaiveBayes Models
# Tasks Left
# Plots in 2D
# get the accuracy values for the six way classifier also.
# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, SpatialDropout1D, GRU, Embedding
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
from keras.preprocessing import sequence, text

#Functions
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

# Dataset Preprocessing
# Output (True:1, False:0)
# y_train_binary = convert_to_bin(y_train)
y_train_binary = np.asarray(y_train)
# y_train_binary = np.reshape(y_train_binary, (10239, 1))
# y_test_binary = convert_to_bin(y_test)
y_test_binary = np.asarray(y_test)
le = preprocessing.LabelEncoder()
y_train_binary = le.fit_transform(y_train_binary)
y_test_binary = le.fit_transform(y_test_binary)

le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
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

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(6, activation='softmax'))

# print(model.summary())
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model = load_model('gru_six.h5')
# model.fit(xtrain_pad, y_train,batch_size=64, validation_data=(xtest_pad,y_test), epochs=10, verbose=1, callbacks=[earlystop])

# Ensemble
print("Ensemble")
pred1_test = logmodel.predict(xtest_ctv)
print(accuracy_score(pred1_test, y_test_binary))

pred2_test = nbmodel.predict(xtest_ctv)
print(accuracy_score(pred2_test, y_test_binary))

pred3_test = model.predict(xtest_pad)

pred3_class = []
for i in pred3_test:
    i = list(i)
    # print(i)
    val = i.index(max(i))
    pred3_class.append(val)
print(accuracy_score(pred3_class, y_test_binary))

next_pred_test = []
for i in range(0, len(y_test_binary)):
    # print(pred1[i], pred2[i], pred3_bin[i])
    data = [pred1_test[i], pred2_test[i], pred3_class[i]]
    # print(data)
    try:
        val = mode(data)
    except:
        val = pred1_test[i]
    next_pred_test.append(val)

next_pred_test = np.asarray(next_pred_test)
print(accuracy_score(next_pred_test, y_test_binary))

# Mode Counting without sentiment
# Individual
# 0.25908372827804105
# 0.2519747235387046
# 0.2282780410742496
# Stacked
# 0.2527646129541864
