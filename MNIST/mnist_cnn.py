# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:11:09 2016

@author: vijay
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas as pd
###############################################################################

def cnn_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(64, 4, 4, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(16, 2, 2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return model

###############################################################################

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

train=df_train.as_matrix(columns=None)
labels=train[:,0].flatten()
train=train[:,1:]
train=train.reshape(train.shape[0],1,28,28).astype('float32')

test=df_test.as_matrix(columns=None)
test=test.reshape(test.shape[0],1,28,28).astype('float32')

print train.shape
print test.shape

###############################################################################

train = train / 255
test = test / 255
labels = np_utils.to_categorical(labels)

# build the model
model = cnn_model()
# Fit the model
model.fit(train, labels,  nb_epoch=10, batch_size=400, verbose=2)
# Final evaluation of the model
scores = model.evaluate(train, labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
predictions = model.predict(test)

res=np.argmax(predictions, axis=1)
np.save('results_adam.npy',res)
