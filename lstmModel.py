import preprocess
from numpy import concatenate
# from matplotlib import pyplot
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding, TimeDistributed, Dropout

preprocess.createNumFeatures()

dataset = read_csv('featureVector.csv', header=0)
val = dataset.values
val=val.astype('float32')
val = np.nan_to_num(val)

X = val[:,:32]
Y = val[:,32]

# print X.shape, Y.shape

X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print X.shape

model = Sequential()
model.add(LSTM(100, input_shape=(32, 1)))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax')) #7 class classification.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(X, Y, epochs=5, batch_size=32, validation_split = 0.2, verbose=1)

model.summary()