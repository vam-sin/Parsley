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

# Functions
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

def binary_classifier():
    train = pd.read_csv('dataset/train2.tsv', delimiter='\t', encoding='utf-8')
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
    x_train = train.iloc[:, [3,15]]
    x_train = np.asarray(x_train)
    x_train = x_train.tolist()
    X = []
    for d in x_train:
        d = str(d[0])
        d = d + str(d[1])
        X.append(d)

    y_train = train.iloc[:,2:3]
    y_train = np.asarray(y_train)
    f = open("statement.txt", "r")
    statement = f.read()
    f = open("justification.txt", "r")
    justification = f.read()
    sample_text = statement+justification
    # Input Word Embeddings
    ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), stop_words = 'english')
    ctv.fit(list(X) + list(X_test))
    sample = []
    sample.append(str(sample_text))
    xtrain_ctv =  ctv.transform(X)
    xtest_ctv = ctv.transform(X_test)
    xsample = ctv.transform(sample)
    y_train_binary = convert_to_bin(y_train)
    y_train_binary = np.asarray(y_train_binary)
    # Output (True:1, False:0)
    le = preprocessing.LabelEncoder()
    y_train_binary = le.fit_transform(y_train_binary)
    print("Logistic Regression")
    logmodel = LogisticRegression()
    logmodel.fit(xtrain_ctv, y_train_binary)

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
    xsample_seq = token.texts_to_sequences(sample)

    # zero pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xtest_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
    xsample_pad = sequence.pad_sequences(xsample_seq, maxlen=max_len)
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
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    nn = load_model('gru_bin.h5')

    # Ensemble
    print("Ensemble")
    pred1_test = logmodel.predict(xsample)
    pred2_test = nbmodel.predict(xsample)
    pred3_test = nn.predict(xsample_pad)

    pred3_bin = []
    for i in pred3_test:
        if i >= 0.55:
            pred3_bin.append(1)
        else:
            pred3_bin.append(0)

    data = [pred1_test[0], pred2_test[0], pred3_bin[0]]
    output = mode(data)
    labels = list(le.inverse_transform([0, 1]))
    output_file = open("binary_output.txt","w")
    output_file.write("Output is " + str(output))
    output_file.write("\n")
    output_file.write("0,1 correspond to " + str(labels) + " respectively")
    output_file.close()

# File Input
binary_classifier()
# Output File
