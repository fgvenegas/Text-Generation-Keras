## Soon
from keras.layers import GRU, Dense, Flatten, Conv1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from numpy import array, argmax
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Utils

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

def parse(text):
  clean_text = re.sub('\ +', ' ', text.replace('\n', ' ').replace('\r', ' '))
  sigma      = set(text)
  
  return clean_text, sigma

def encode_to_text(encode, label_encoder):
    text = ""

    for i in encode:
        inverted = label_encoder.inverse_transform([argmax(i)])
        text += inverted[0]
  
    return text


import re

INPUT_FILEPATH = 'quijote2.txt'
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


# Model 
model = Sequential()
model.add(Conv1D(filters=150, kernel_size=(3,), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(128, return_sequences=True))
model.add(GRU(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(sigma), activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


#from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True)

model.fit(X_train, y_train, epochs=14, batch_size=32, callbacks=callbacks_list)

model.save_weights('my_model_weights.h5')