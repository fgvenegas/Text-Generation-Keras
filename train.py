
# coding: utf-8

# # Text Generation

# In[20]:

import keras

from keras.layers import GRU, Dense, Flatten, Conv1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from numpy import array, argmax
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# In[24]:

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ## Utils

# In[2]:

def text_to_encode(text, alphabet):
    values = array(list(alphabet))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    text = list(text)
    encoded_phrase = onehot_encoder.transform(label_encoder.transform(text).reshape(len(text), 1))
    return encoded_phrase, label_encoder, onehot_encoder

def encode_to_text(encode, label_encoder):
    text = ""

    for i in encode:
        inverted = label_encoder.inverse_transform([argmax(i)])
        text += inverted[0]
  
    return text

def parse(text):
    clean_text = re.sub('\ +', ' ', text.replace('\n', ' ').replace('\r', ' '))
    sigma      = set(text)
  
    return clean_text, sigma


# ## Training set

# In[3]:

import re

INPUT_FILEPATH = 'chat.txt'
WINDOW_SIZE    = 100

X_train = None; y_train = None

with open(INPUT_FILEPATH, 'r') as input_file:
    text, sigma = parse(input_file.read())
  
    X_train = np.zeros((len(text) - WINDOW_SIZE + 1, WINDOW_SIZE - 1, len(sigma)))
    y_train = np.zeros((len(text) - WINDOW_SIZE + 1, 1, len(sigma)))
  
    encoded_text, label_encoder, onehot_encoder = text_to_encode(text, sigma)
  
    i = 0
    while i + WINDOW_SIZE < len(encoded_text) + 1:
        X_train[i] = encoded_text[i:i + WINDOW_SIZE - 1]
        y_train[i] = encoded_text[i + WINDOW_SIZE - 1]
        i += 1

# Ejemplos de correctitud
# print(text_train_X[0], len(text_train_X[0]), text_train_y[0], len(text_train_y[0]))
# print(text_train_X[-1], len(text_train_X[-1]), text_train_y[-1], len(text_train_y[-1]))


# In[4]:

y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])


# In[5]:

X_train.shape, y_train.shape


# In[13]:

encode_to_text(X_train[111], label_encoder)


# ## Model

# In[16]:

model = Sequential()
model.add(Conv1D(filters=150, kernel_size=(3,), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(256, return_sequences=True))
model.add(GRU(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(sigma), activation='softmax'))
model.summary()


# In[18]:

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )


# In[21]:

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[27]:

model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=callbacks_list)


# In[28]:

model.save_weights('my_model_weights.h5')


