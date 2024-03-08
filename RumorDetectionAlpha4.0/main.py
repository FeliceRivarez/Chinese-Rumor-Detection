import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten,MaxPooling1D
import tensorflow as tf
import torch
from keras_preprocessing import sequence
from sklearn.svm import SVC
from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import SimpleRNN, Dropout, LSTM, Bidirectional, GlobalMaxPooling1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.saved_model.load import load

from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tensorflow.keras import layers

# 加载训练集
train=pd.read_csv('./parsedTrain.csv')
train=train.sample(frac=1)
docs=train['Title'].fillna("")
targets=train["label"]

# verify=pd.read_csv("./labeledVerification.csv")
# docs_veri=verify["Title"].fillna("")
testing=pd.read_csv("./parsedTest.csv")
docs_test=testing["Title"].fillna("")
# verifyAnswer=np.asarray[verify["label"]]

tok = Tokenizer(num_words=10000)
tok.fit_on_texts(docs.values)

vocab_size = 10000#估计的词汇表大小
encoded_docs = tok.texts_to_sequences(docs)
# encoded_verify= tok.texts_to_sequences(docs_veri)
encoded_test=tok.texts_to_sequences(docs_test)


# 将序列数据填充成相同长度
padded_docs= sequence.pad_sequences(encoded_docs, maxlen=60)
# padded_verify= sequence.pad_sequences(encoded_verify, maxlen=60)
padded_test= sequence.pad_sequences(encoded_test, maxlen=60)




padded_docs=np.expand_dims(padded_docs,axis=2)
# padded_verify=np.expand_dims(padded_verify,axis=2)
padded_test=np.expand_dims(padded_test,axis=2)


# 以下模型，需要使用哪个，就把哪个取消注释即可，剩下的注释掉

# LSTM
EMBEDDING_DIM = 32
model = Sequential()
model.add(Embedding(10000, EMBEDDING_DIM))
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# BiLSTM
EMBEDDING_DIM = 128
model = Sequential()
model.add(Embedding(10000, EMBEDDING_DIM))
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 1)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# simpleRNN
EMBEDDING_DIM = 128
model = Sequential()
model.add(Embedding(20000, EMBEDDING_DIM))
model.add(Dropout(0.1))
model.add(SimpleRNN(64))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# CNN-RNN
EMBEDDING_DIM = 128
model = Sequential()
model.add(Embedding(10000, EMBEDDING_DIM))
model.add(Conv1D(filters=32, kernel_size=15, padding="same",activation="relu"))
model.add(MaxPooling1D(pool_size=5,strides=1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(SimpleRNN(32))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 模型设定与编译
cp=ModelCheckpoint('model_Rnn.hdf5',
                   monitor='val_accuracy',
                   verbose=1,
                   save_best_only=True,
                   mode='auto'
                   )

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
print(model.summary())

VALIDATION_SPLIT = 0.2
EPOCHS = 50
history = model.fit(padded_docs,
                    targets,
                    batch_size = 1000,
                    # validation_split = VALIDATION_SPLIT,
                    validation_split=VALIDATION_SPLIT,
                    epochs = EPOCHS,
                    shuffle=True,
                    callbacks=[cp],
                    validation_freq=1,
                    verbose=100
                    )

pred=model.predict(padded_test)

result=open('./result.csv','w')
result.write("id,prob"+'\n')
temp=0
ID=0
true=0
fake=0
for i in pred:
    temp+=1
    if temp%1==0:
        # print(temp)
        result.close()
        result=open('./result.csv','a')
    ID+=1
    result.write(str(ID))
    result.write(',')
    result.write(str(i[0]))
    result.write('\n')
print(true)
print(fake)
print(temp)






